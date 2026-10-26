"""Causal vehicle closing estimates with immediate response to increasing risk."""
from carla_tools.traffic_perception import CausalObjectTracker,_iou


class AdaptiveVehicleTracker(CausalObjectTracker):
    """Retain association, identities, boxes and pedestrian behavior unchanged.

    At exactly two valid vehicle velocity samples, average the middle pair but
    never report a closing speed smaller than the latest sample. Three-sample
    and pedestrian estimates retain the original median selection.
    """
    def update(self,objects,timestamp):
        self.tracks={key:value for key,value in self.tracks.items() if 0<timestamp-value['timestamp']<=self.config.track_max_age_s}
        unmatched=set(self.tracks);result=[]
        for item in sorted(objects,key=lambda r:-r['confidence']):
            matches=[(key,_iou(item['bbox_xyxy'],self.tracks[key]['bbox'])) for key in unmatched
                if self.tracks[key]['label']==item['label']]
            key,overlap=max(matches,key=lambda p:p[1],default=(None,0.))
            if overlap<self.config.track_min_iou:key=None
            if key is None:self.sequence+=1;key=f'traffic_{self.sequence:06d}';past=None
            else:past=self.tracks[key];unmatched.remove(key)
            closing=None;history=[]
            if past and item['distance_m'] is not None and past['distance_m'] is not None:
                dt=timestamp-past['timestamp']
                if dt>=.025:
                    instantaneous=(past['distance_m']-item['distance_m'])/dt
                    if abs(instantaneous)<=30.:
                        history=(past.get('closing_history',[])+[instantaneous])[-3:]
                        if len(history)>=2:
                            closing=sorted(history)[len(history)//2]
                            if item.get('kind')=='vehicle' and len(history)==2:
                                closing=max(history[-1],.5*(history[0]+history[1]))
            item=dict(item,track_id=key,closing_speed_mps=closing)
            self.tracks[key]={'bbox':item['bbox_xyxy'],'label':item['label'],'distance_m':item['distance_m'],
                'timestamp':timestamp,'closing_history':history}
            result.append(item)
        return result
