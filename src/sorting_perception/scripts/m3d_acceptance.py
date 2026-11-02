"""Grade two class batches and two covered fixtures from complete evidence."""
def assess(reports):
    checks={};clear={r.get('expected_class'):r for r in reports if not r.get('expected_pass')};covered={r.get('expected_class'):r for r in reports if r.get('expected_pass')}
    classes={'red_fruit','green_fruit'};checks['four_distinct_reports']=len(reports)==4 and set(clear)==classes and set(covered)==classes
    checks['same_validated_install']=bool(reports) and all(r.get('installed_manifest') and r['installed_manifest']==reports[0].get('installed_manifest') for r in reports)
    checks['evaluator_isolation']=all(r.get('checkpoint')=='M3D' and r.get('status')=='PASS' and r.get('isolation_audit',{}).get('status')=='PASS' for r in reports)
    checks['safe_final_reset']=all(r.get('final_reset_status')=='PASS' and r.get('final_owner')=='HOLD' and r.get('overlap_violations')==0 and not r.get('stopped_early') for r in reports)
    checks['all_trials_reset']=all(r.get('trials') and all(t.get('reset',{}).get('status')=='PASS' for t in r['trials']) for r in reports)
    checks['complete_class_batches']=all(len(r.get('trials',[]))==10 and r.get('completed_trials')==10 and r.get('aborted_trials')==0 and sum(t.get('status')=='PASS' for t in r['trials'])>=8 for r in clear.values()) and len(clear)==2
    checks['covered_passes']=all(len(r.get('trials',[]))==1 and r['trials'][0].get('status')=='PASS' and r['trials'][0].get('physical',{}).get('action')=='PASS' for r in covered.values()) and len(covered)==2
    checks['correct_class_and_bin']=all(t.get('decision',{}).get('class_id')==cls and t.get('physical',{}).get('selected_bin')==('bin_a' if cls=='red_fruit' else 'bin_b') for cls,r in clear.items() for t in r.get('trials',[]) if t.get('status')=='PASS')
    success=sum(t.get('status')=='PASS' for r in clear.values() for t in r.get('trials',[]));checks['at_least_16_of_20']=success>=16
    passed=all(checks.values());return dict(checkpoint='M3D',status='PASS' if passed else 'FAIL',full_M3_gate_passed=passed,successful_sorts=success,nominal_trials=20,checks={k:'PASS' if v else 'FAIL' for k,v in checks.items()},scope='Fixed prepositioned wrist, synthetic RGB classes, nominal 0.03 m/s. Active inspection and speed/delay stress tests remain later work.')
