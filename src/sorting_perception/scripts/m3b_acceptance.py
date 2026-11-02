"""Acceptance uses complete trial/reset evidence, including a failed last trial."""
def accepted_batch(trials,requested,minimum,final_reset,final_owner,overlap,stopped_early):
    threshold=1 if requested==1 else max(16,int(minimum))
    complete=(len(trials)==requested and all(t.get('status') in {'PASS','FAIL'} for t in trials))
    resets=complete and all(t.get('reset',{}).get('status')=='PASS' for t in trials)
    return bool(not stopped_early and complete and resets and sum(t['status']=='PASS' for t in trials)>=threshold and
                final_reset=='PASS' and final_owner=='HOLD' and overlap==0)
