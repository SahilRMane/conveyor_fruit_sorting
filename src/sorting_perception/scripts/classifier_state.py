"""Exact-capture score/target matching with reset and age guards."""
import copy,math
from classifier_core import require

def ns(t):return t.sec*1_000_000_000+t.nanosec
class EvidenceStore:
    def __init__(self,c):self.c=c;self.episode=None;self.barrier=0;self.scores={}
    def reset(self,episode,stamp):
        if self.episode is not None and episode<=self.episode:return False
        self.episode=episode;self.barrier=stamp;self.scores.clear();return True
    def add(self,m):
        capture=ns(m.header.stamp)
        if not m.valid or m.episode_id!=self.episode or capture<=self.barrier:return False
        p=list(m.probabilities)
        require(list(m.class_ids)==self.c['class_ids'] and len(p)==len(m.class_ids) and all(math.isfinite(v) and 0<=v<=1 for v in p) and abs(sum(p)-1)<1e-6,'invalid_class_distribution')
        self.scores[(m.episode_id,m.view_id,capture)]=copy.deepcopy(m)
        while len(self.scores)>8:self.scores.pop(next(iter(self.scores)))
        return True
    def enrich(self,message,now):
        out=copy.deepcopy(message)
        for b in out.targets:
            b.class_ids=[];b.class_probabilities=[]
            key=(b.episode_id,b.view_id,ns(b.header.stamp));score=self.scores.get(key)
            if (b.valid and b.camera_only and b.episode_id==self.episode and ns(b.header.stamp)>self.barrier and score is not None and
                -.02<=(now-ns(score.header.stamp))/1e9<self.c['maximum_capture_age_s']):
                b.class_ids=list(score.class_ids);b.class_probabilities=list(score.probabilities)
        return out
