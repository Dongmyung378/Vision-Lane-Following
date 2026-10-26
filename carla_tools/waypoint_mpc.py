"""Bounded waypoint steering MPC with batched CPU rollouts and PP longitudinal PI.

No route, world pose, GT lane state, learned pose head, or lane width is read.
The measured-speed constant-velocity rollout is a model, not a safety proof.
"""
from dataclasses import dataclass,fields,asdict
import copy
import math

from carla_tools.contracts import canonical_sha256
from carla_tools.controller import ControllerError,VehicleGeometry,_rear_points
from carla_tools.controller_study import ControllerStudyConfig,build_controller

VERSION='bounded-RGB-far-waypoint-kinematic-steering-MPC-v1'

def require(value,code):
    if not value:raise ControllerError(code)

def clip(value,low,high):return max(low,min(high,value))

@dataclass(frozen=True)
class MPCConfig:
    schema_version:int=1
    controller_version:str=VERSION
    horizon_seconds:float=2.
    prediction_steps:int=40
    offsets:tuple=(-.04,-.02,0.,.02,.04)
    far_horizons_m:tuple=(6.,10.,15.)
    minimum_supported_steps:int=4
    lateral_squared_weight:float=1.
    heading_squared_weight:float=2.
    steering_change_weight:float=.05
    steering_effort_weight:float=.001
    strict_cost_improvement_tolerance:float=1e-10

    def __post_init__(self):
        require(self.schema_version==1 and self.controller_version==VERSION
            and self.horizon_seconds==2. and type(self.prediction_steps) is int and self.prediction_steps==40
            and self.offsets==(-.04,-.02,0.,.02,.04) and self.far_horizons_m==(6.,10.,15.)
            and self.minimum_supported_steps==4 and self.lateral_squared_weight==1.
            and self.heading_squared_weight==2. and self.steering_change_weight==.05
            and self.steering_effort_weight==.001 and self.strict_cost_improvement_tolerance==1e-10,
            'MPC-fixed-config-required')

    def to_dict(self):
        result=asdict(self)
        for key in ('offsets','far_horizons_m'):result[key]=list(result[key])
        return result

    @classmethod
    def from_dict(cls,value):
        require(type(value) is dict and set(value)=={f.name for f in fields(cls)},'MPC-config-fields')
        copy_value=dict(value)
        for key in ('offsets','far_horizons_m'):copy_value[key]=tuple(copy_value[key])
        return cls(**copy_value)

    @property
    def sha256(self):return canonical_sha256(self.to_dict())

def rear_curvature(normalized_steer,speed,geometry):
    """PhysX normalized inner-wheel command -> rear bicycle curvature."""
    limit=math.radians(geometry.max_steer_angle_deg)*geometry.steering_factor(speed)
    angle=abs(normalized_steer)*limit;tangent=math.tan(angle)
    denominator=geometry.wheelbase_m+.5*geometry.front_track_m*tangent
    require(math.isfinite(tangent) and denominator>0.,'MPC-invalid-measured-steering-curvature')
    return math.copysign(tangent/denominator,normalized_steer) if normalized_steer else 0.

def far_reference(waypoints,geometry):
    """Sample6/10/15 using predicted segment lengths, never GT arc or x.

    Y and X individually need not be monotone in a turn. The existing PP
    contract requires finite, rear-forward points with increasing rear radius.
    No origin-to-path connector or backward current-lane extrapolation is made.
    """
    _rear_points(waypoints,geometry.rear_axle_x_m,geometry.rear_axle_y_m)
    points=[[float(x),float(y)] for x,y in waypoints]
    distances=[math.hypot(*points[0])]
    for first,last in zip(points,points[1:]):
        length=math.hypot(last[0]-first[0],last[1]-first[1])
        require(length>1e-6,'MPC-degenerate-predicted-segment')
        distances.append(distances[-1]+length)
    require(distances[0]<=6. and distances[-1]>10.+1e-6,'MPC-insufficient-predicted-far-support')
    actual_targets=[6.,10.,min(15.,distances[-1])];selected=[];segments=[]
    for target in actual_targets:
        index=next(i for i in range(4) if distances[i]<=target<=distances[i+1]+1e-9)
        alpha=clip((target-distances[index])/(distances[index+1]-distances[index]),0.,1.)
        selected.append([points[index][axis]+alpha*(points[index+1][axis]-points[index][axis]) for axis in range(2)])
    for first,last in zip(selected,selected[1:]):
        dx,dy=last[0]-first[0],last[1]-first[1];length=dx*dx+dy*dy
        require(length>1e-8,'MPC-degenerate-far-reference')
        segments.append((first[0],first[1],dx,dy,length,math.atan2(dy,dx)))
    return selected,segments,{'distance_index':'predicted-polyline;first-anchor-norm-for-index-only',
        'predicted_support_m':[distances[0],distances[-1]],'requested_far_distances_m':[6.,10.,15.],
        'supported_far_distances_m':actual_targets,'last_target_clamped_to_prediction_end':distances[-1]<15.,
        'origin_connector_used_as_lane_reference':False,'GT_arc_used':False,'forward_x_monotonicity_required':False}

def path_cost(actor_x,actor_y,yaw,segments):
    best=None
    for px,py,dx,dy,length,heading in segments:
        fraction=clip(((actor_x-px)*dx+(actor_y-py)*dy)/length,0.,1.)
        dx_error=actor_x-px-fraction*dx;dy_error=actor_y-py-fraction*dy
        distance_squared=dx_error*dx_error+dy_error*dy_error
        if best is None or distance_squared<best[0]:
            angle=math.atan2(math.sin(yaw-heading),math.cos(yaw-heading))
            best=(distance_squared,angle*angle)
    return best

def rollout(knots,speed,previous,geometry,base_config,mp_config,segments,supported_indices):
    dt=mp_config.horizon_seconds/mp_config.prediction_steps
    rx,ry=geometry.rear_axle_x_m,geometry.rear_axle_y_m
    x,y,yaw,steer=rx,ry,0.,previous;geometric_cost=0.;change_cost=0.;effort=0.
    first=None;maximum_lateral=0.;terminal=None;support=set(supported_indices)
    for index in range(mp_config.prediction_steps):
        desired=knots[0 if index<mp_config.prediction_steps//2 else 1]
        desired=clip(desired,-base_config.max_normalized_steer,base_config.max_normalized_steer)
        filtered=steer+base_config.steer_smoothing_alpha*(desired-steer)
        new_steer=clip(filtered,steer-base_config.steer_rate_per_second*dt,steer+base_config.steer_rate_per_second*dt)
        change_cost+=(new_steer-steer)**2;steer=new_steer;effort+=steer*steer
        if first is None:first=steer
        curvature=rear_curvature(steer,speed,geometry)
        rear_speed=speed/math.sqrt((1.+ry*curvature)**2+(rx*curvature)**2)
        new_yaw=yaw+rear_speed*curvature*dt
        if abs(curvature)>1e-9:
            x+=(math.sin(new_yaw)-math.sin(yaw))/curvature
            y+=(-math.cos(new_yaw)+math.cos(yaw))/curvature
        else:x+=rear_speed*dt*math.cos(yaw);y+=rear_speed*dt*math.sin(yaw)
        yaw=new_yaw;actor_x=x-rx*math.cos(yaw)+ry*math.sin(yaw);actor_y=y-rx*math.sin(yaw)-ry*math.cos(yaw)
        maximum_lateral=max(maximum_lateral,rear_speed*rear_speed*abs(curvature))
        if index in support:
            distance,heading=path_cost(actor_x,actor_y,yaw,segments)
            geometric_cost+=mp_config.lateral_squared_weight*distance+mp_config.heading_squared_weight*heading
        terminal=[actor_x,actor_y,yaw]
    require(support,'MPC-no-unsupported-zero-cost-rollout')
    total=(geometric_cost/len(support)+mp_config.steering_change_weight*change_cost/mp_config.prediction_steps
        +mp_config.steering_effort_weight*effort/mp_config.prediction_steps)
    require(all(math.isfinite(v) for v in (total,first,maximum_lateral)),'MPC-nonfinite-rollout')
    return {'cost':total,'first_steer':first,'terminal_actor_xy_yaw':terminal,
        'maximum_model_rear_lateral_acceleration_mps2':maximum_lateral,
        'supported_steps':len(support),'steering_change_squared_sum':change_cost}

def vector_candidate_rollouts(values,speed,previous,geometry,base_config,mp_config,segments,supported):
    """Exact alpha1/slew semantics, batching25 candidates and40 time points."""
    import numpy as np
    require(base_config.steer_smoothing_alpha==1.,'MPC-vector-alpha1-required')
    dt=mp_config.horizon_seconds/mp_config.prediction_steps
    values=np.asarray(values,dtype=np.float64)
    a=np.repeat(values,5);b=np.tile(values,5);step=base_config.steer_rate_per_second*dt
    ramp=np.arange(1,21,dtype=np.float64)[:,None]*step
    first=previous+np.clip(a[None,:]-previous,-ramp,ramp)
    second=first[-1]+np.clip(b[None,:]-first[-1],-ramp,ramp)
    steer=np.concatenate((first,second),axis=0)
    tangent=np.tan(np.abs(steer)*math.radians(geometry.max_steer_angle_deg)*geometry.steering_factor(speed))
    curvature=np.sign(steer)*tangent/(geometry.wheelbase_m+.5*geometry.front_track_m*tangent)
    rx,ry=geometry.rear_axle_x_m,geometry.rear_axle_y_m
    rear_speed=speed/np.sqrt((1.+ry*curvature)**2+(rx*curvature)**2)
    increments=rear_speed*curvature*dt;yaw=np.cumsum(increments,axis=0)
    before=np.vstack((np.zeros((1,25)),yaw[:-1]))
    dx=rear_speed*dt*np.cos(before);dy=rear_speed*dt*np.sin(before)
    np.divide(np.sin(yaw)-np.sin(before),curvature,out=dx,where=np.abs(curvature)>1e-9)
    np.divide(-np.cos(yaw)+np.cos(before),curvature,out=dy,where=np.abs(curvature)>1e-9)
    rear_x=rx+np.cumsum(dx,axis=0)
    rear_y=ry+np.cumsum(dy,axis=0)
    actor_x=rear_x-rx*np.cos(yaw)+ry*np.sin(yaw)
    actor_y=rear_y-rx*np.sin(yaw)-ry*np.cos(yaw)
    indices=np.asarray(supported,dtype=np.int64)
    require(len(indices)>0,'MPC-no-unsupported-zero-cost-rollout')
    reference=np.asarray(segments,dtype=np.float64)
    dx=actor_x[indices,:,None]-reference[None,None,:,0]
    dy=actor_y[indices,:,None]-reference[None,None,:,1]
    fraction=np.clip((dx*reference[None,None,:,2]+dy*reference[None,None,:,3])/reference[None,None,:,4],0.,1.)
    squared=(dx-fraction*reference[None,None,:,2])**2+(dy-fraction*reference[None,None,:,3])**2
    nearest=np.argmin(squared,axis=2)
    dist=np.take_along_axis(squared,nearest[:,:,None],axis=2)[:,:,0]
    tangent_heading=reference[:,5][nearest]
    angle=yaw[indices]-tangent_heading;heading=np.arctan2(np.sin(angle),np.cos(angle))
    geometric=np.mean(mp_config.lateral_squared_weight*dist+mp_config.heading_squared_weight*heading**2,axis=0)
    delta=steer-np.vstack((np.full((1,25),previous),steer[:-1]))
    change=np.sum(delta**2,axis=0);effort=np.sum(steer**2,axis=0)
    total=geometric+mp_config.steering_change_weight*change/40.+mp_config.steering_effort_weight*effort/40.
    lateral=np.max(rear_speed**2*np.abs(curvature),axis=0)
    require(bool(np.isfinite(total).all()) and bool(np.isfinite(lateral).all()),'MPC-vector-nonfinite-rollout')
    return [{'knots':[float(a[i]),float(b[i])],'cost':float(total[i]),'first_steer':float(steer[0,i]),
        'terminal_actor_xy_yaw':[float(actor_x[-1,i]),float(actor_y[-1,i]),float(yaw[-1,i])],
        'maximum_model_rear_lateral_acceleration_mps2':float(lateral[i]),
        'supported_steps':len(supported),'steering_change_squared_sum':float(change[i])} for i in range(25)]


class MPCController:
    def __init__(self,base_study_config,geometry,mp_config=None):
        require(isinstance(base_study_config,ControllerStudyConfig) and base_study_config.method=='pure_pursuit'
            and isinstance(geometry,VehicleGeometry),'MPC-original-PP-and-measured-geometry-required')
        self.config=base_study_config;self.geometry=geometry;self.mp_config=mp_config or MPCConfig()
        require(isinstance(self.mp_config,MPCConfig),'MPC-validated-fixed-config-required')
        require(abs(self.config.fixed_delta_seconds-.05)<1e-12 and abs(self.config.steer_smoothing_alpha-1.)<1e-12,
            'MPC-original50ms-alpha1-policy-required')
        self._pp=build_controller(base_study_config,geometry);self.reset()

    def reset(self):self._pp.reset();self._previous_steer=0.;self._step=0

    def step(self,waypoints,speed_mps,dt_seconds=.05):
        require(type(speed_mps) in (int,float) and math.isfinite(speed_mps) and 0<=speed_mps<=100.,'MPC-finite-forward-speed')
        require(abs(dt_seconds-.05)<1e-12,'MPC-fixed50ms-step')
        previous=self._previous_steer
        # PP's PI advances once; all steering proposals share ACTUAL history.
        self._pp._previous_steer=previous
        pp=self._pp.step(waypoints,speed_mps,dt_seconds=dt_seconds)
        request=pp['diagnostics']['steer_clipped'];baseline_first=pp['steer']
        selected=None;baseline=None;baseline_knots=[request,request];selected_knots=baseline_knots
        supported=[i for i in range(40) if speed_mps*dt_seconds*(i+1)>=6.]
        fallback=None;points=None;candidates=[];reference_support=None
        if speed_mps*2.<6.:fallback='constant-speed2s-horizon-cannot-reach6m'
        elif len(supported)<self.mp_config.minimum_supported_steps:fallback='fewer-than4-far-supported-rollout-steps'
        else:
            try:
                points,segments,reference_support=far_reference(waypoints,self.geometry)
                values=[clip(request+offset,-self.config.max_normalized_steer,self.config.max_normalized_steer)
                    for offset in self.mp_config.offsets]
                candidates=vector_candidate_rollouts(values,speed_mps,previous,self.geometry,self.config,
                    self.mp_config,segments,supported)
                # Central offset0/0 is exactly the original PP baseline candidate.
                baseline=candidates[12]
                require(abs(baseline['first_steer']-baseline_first)<=1e-12,'MPC-PP-first-command-equivalence')
                selected=baseline
                for trial in candidates:
                    if trial['cost']<selected['cost']-self.mp_config.strict_cost_improvement_tolerance:
                        selected=trial;selected_knots=trial['knots']
            except (ControllerError,ValueError,OverflowError,ZeroDivisionError) as error:
                fallback='invalid-or-infeasible-far-reference:'+type(error).__name__;selected=None
        applied=baseline_first if selected is None else selected['first_steer']
        require(abs(applied)<=self.config.max_normalized_steer+1e-12
            and abs(applied-previous)<=self.config.steer_rate_per_second*dt_seconds+1e-12,'MPC-real-actuator-bounds')
        command=copy.deepcopy(pp);command['steer']=applied
        detail={'version':VERSION,'active':selected is not None,'fallback_reason':fallback,
            'config_sha256':self.mp_config.sha256,'prediction_horizon_seconds':2.,'prediction_steps':40,
            'far_reference_nominal_arc_m':[6.,10.,15.],'far_reference_actor_xy_m':points,
            'far_reference_support':reference_support,
            'rollout_support_steps':len(supported),'original_PP_request':request,'original_PP_first_steer':baseline_first,
            'previous_actual_steer':previous,'first_applied_steer':applied,'PP_baseline_candidate_always_included':True,
            'PP_baseline_knots':baseline_knots,'baseline_PP_cost':baseline['cost'] if baseline else None,
            'selected_cost':selected['cost'] if selected else None,'selected_knots':selected_knots,
            'candidate_count':len(candidates) if candidates else 1,'candidates':candidates,
            'maximum_selected_model_rear_lateral_acceleration_mps2':selected['maximum_model_rear_lateral_acceleration_mps2'] if selected else None,
            'lateral_acceleration_is_diagnostic_not_true_road_or_tyre_constraint':True,
            'GT_inputs':False,'current_pose_head_used':False,'longitudinal_policy_changed':False,
            'actual_smoothing_and_slew_applied_once':True,'constant_speed_model':True,
            'physical_lane_corridor_guaranteed':False,'reaches_unobserved_near_lane_claimed':False}
        command['diagnostics'].update(controller_version=VERSION,controller_method='bounded-RGB-waypoint-kinematic-MPC',
            rgb_waypoint_mpc=detail,step_index=self._step,steer_before_smoothing=applied,steer_filtered=applied,
            steering_diagnostics_scope='original-PP-path-diagnostics-plus-explicit-MPC-actual-command',
            steer_rate_limited=abs(applied-previous)>=self.config.steer_rate_per_second*dt_seconds-1e-12)
        self._previous_steer=applied;self._pp._previous_steer=applied;self._step+=1
        return command
