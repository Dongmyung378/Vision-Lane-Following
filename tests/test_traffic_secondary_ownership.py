"""Synthetic ownership and lifecycle contracts; no model or simulator."""
import copy
import subprocess
import sys
import unittest
from carla_tools import traffic_secondary_ownership as owned
from carla_tools.traffic_secondary_veto import SecondaryPersonVetoBridge
import test_traffic_secondary_person as peer

def p(key,box,score=.7,**extras):
    return dict(kind='pedestrian',track_id=key,bbox_xyxy=box,confidence=score,**extras)

def d(box,score=.9,label=1):return dict(bbox_xyxy=box,score=score,label_id=label)

def raw(items):return dict(GT_inputs=False,objects=copy.deepcopy(items))


class ExclusiveOwnershipTests(unittest.TestCase):
    def test_unique_person_keeps_its_positive(self):
            owners,ambiguous,_=owned.assign_secondary_people(raw([p('a',[10.,10.,30.,50.])]),[d([10.,10.,30.,50.])],100,100)
            self.assertEqual(len(owners['a']),1);self.assertFalse(ambiguous)

    def test_two_distinct_disjoint_people_each_keep_separate_evidence(self):
            boxes=[[10.,10.,30.,50.],[60.,10.,80.,50.]]
            owners,ambiguous,decisions=owned.assign_secondary_people(raw([p(str(i),b) for i,b in enumerate(boxes)]),[d(b) for b in boxes],100,100)
            self.assertEqual([len(owners[str(i)]) for i in range(2)],[1,1]);self.assertFalse(ambiguous)
            self.assertEqual([x['owner_primary_track_id'] for x in decisions],['0','1'])

    def test_two_distinct_overlapping_people_keep_their_own_best_boxes(self):
            boxes=[[10.,10.,30.,50.],[14.,10.,34.,50.]]
            owners,ambiguous,_=owned.assign_secondary_people(raw([p(str(i),b) for i,b in enumerate(boxes)]),[d(b) for b in boxes],100,100)
            self.assertEqual([len(owners[str(i)]) for i in range(2)],[1,1]);self.assertFalse(ambiguous)

    def test_merged_secondary_for_two_real_people_remains_ambiguous(self):
            boxes=[[10.,10.,30.,50.],[14.,10.,34.,50.]]
            owners,ambiguous,decisions=owned.assign_secondary_people(raw([p(str(i),b) for i,b in enumerate(boxes)]),[d([12.,10.,32.,50.])],100,100)
            self.assertEqual(ambiguous,{'0','1'});self.assertFalse(any(owners.values()))
            self.assertIsNone(decisions[0]['owner_primary_track_id'])

    def test_identical_boxes_do_not_resolve_by_track_name_or_confidence(self):
            box=[10.,10.,30.,50.]
            for items in ([p('z',box,.99),p('a',box,.5)],[p('a',box,.5),p('z',box,.99)]):
                owners,ambiguous,_=owned.assign_secondary_people(raw(items),[d(box)],100,100)
                self.assertEqual(ambiguous,{'a','z'});self.assertFalse(any(owners.values()))

    def test_strong_ineligible_current_primary_still_owns_its_evidence(self):
            items=[p('strong',[10.,10.,30.,50.],.9,ground_valid=False,in_path=False),p('weak',[12.,15.,20.,45.])]
            owners,_,_=owned.assign_secondary_people(raw(items),[d([10.,10.,30.,50.])],100,100)
            self.assertEqual(len(owners['strong']),1);self.assertFalse(owners['weak'])

    def test_two_duplicate_secondary_boxes_never_fall_back_to_weaker_subbox(self):
            items=[p('strong',[10.,10.,30.,50.],.9),p('weak',[12.,15.,20.,45.])]
            owners,_,decisions=owned.assign_secondary_people(raw(items),[d([10.,10.,30.,50.]),d([10.,10.,30.,50.],.8)],100,100)
            self.assertEqual(len(owners['strong']),2);self.assertEqual(owners['weak'],[])
            self.assertTrue(all(not x['fallback_reassignment_performed'] for x in decisions))

    def test_nonperson_and_low_score_secondary_are_ignored(self):
            box=[10.,10.,30.,50.]
            owners,_,decisions=owned.assign_secondary_people(raw([p('p',box)]),[d(box,.499),d(box,.99,10),d(box,.99,3)],100,100)
            self.assertEqual(owners['p'],[]);self.assertEqual(decisions,[])

    def test_assignment_never_mutates_primary_or_reads_GT(self):
            source=raw([p('p',[10.,10.,30.,50.])]);old=copy.deepcopy(source)
            owned.assign_secondary_people(source,[d([10.,10.,30.,50.])],100,100)
            self.assertEqual(source,old)
            source['GT_inputs']=True
            with self.assertRaisesRegex(ValueError,'no-GT'):
                owned.assign_secondary_people(source,[],100,100)

    def test_import_starts_no_neural_runtime(self):
        subprocess.run([sys.executable, "-c", "import sys; import carla_tools.traffic_secondary_ownership; assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules"], check=True, capture_output=True)

    def test_scheduler_tie_None_does_not_latch_positive(self):
        first=peer.person();second=peer.person("other")
        backend=peer.RecordedFixtureBackend([d(first["bbox_xyxy"])])
        scheduler=owned.ExclusiveSecondaryPersonConfirmationScheduler(backend,peer.calibration(),primary_source_identity=peer.PRIMARY)
        packet=scheduler.update(peer.raw(0,[first,second]),peer.observation(0))
        self.assertEqual([c["secondary_person_confirmation"]for c in packet["confirmations"]],[None,None])
        self.assertFalse(any(t["positive_confirmed"]for t in scheduler.tracks.values()))

    def test_unique_positive_dropout_preserves_existing_memory(self):
        person=peer.person();backend=peer.RecordedFixtureBackend([d(person["bbox_xyxy"])])
        scheduler=owned.ExclusiveSecondaryPersonConfirmationScheduler(backend,peer.calibration(),primary_source_identity=peer.PRIMARY)
        bridge=SecondaryPersonVetoBridge(nominal_calibration=peer.calibration(),secondary_scheduler=scheduler)
        for i in range(16):result,notes=peer.step(bridge,i,[person]if i==0 else [])
        self.assertIn("person",bridge.secondary_veto_ids);self.assertIn("person",bridge.tracks)
        self.assertFalse(result["perception_valid"]);self.assertEqual(backend.calls,1)

    def test_current_positive_lateral_clearance_keeps_original_hold(self):
        person=peer.person();backend=peer.RecordedFixtureBackend([d(person["bbox_xyxy"])])
        scheduler=owned.ExclusiveSecondaryPersonConfirmationScheduler(backend,peer.calibration(),primary_source_identity=peer.PRIMARY)
        bridge=SecondaryPersonVetoBridge(nominal_calibration=peer.calibration(),secondary_scheduler=scheduler)
        peer.step(bridge,0,[person]);outside=peer.person(bbox_xyxy=[500.,136.,518.,169.],lateral_y_m=8.,in_path=False)
        peer.step(bridge,1,[outside]);self.assertIn("person",bridge.tracks)
        for i in range(2,8):peer.step(bridge,i,[outside])
        self.assertNotIn("person",bridge.tracks);self.assertNotIn("person",bridge.secondary_veto_ids)

if __name__=="__main__":unittest.main()
