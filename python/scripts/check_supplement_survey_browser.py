"""Exercise the retained survey and new evidence controls in an actual browser."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'develop_logs/test_runs/20261028_supplement/survey_browser'
OUT.mkdir(exist_ok=True)
errors=[];checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:8876/develop_logs/notes/motiongen_survey_revision/dist/index.html',wait_until='networkidle',timeout=60000)
    for method in ['motionlcm','omnicontrol','chois','ardy','kimodo','dart','primal','closd']:
        page.locator('#native-method-filter').select_option(method)
        for kind in ['backbone','loop']:
            svg=page.locator(f'svg[aria-label="{method} {kind}"]')
            assert svg.count()==1
            svg.screenshot(path=str(OUT/f'{method}_{kind}.png'))
        if page.locator('#paired-motion-play').count():
            timer=page.locator('#paired-motion-time')
            before=timer.input_value();page.locator('#paired-motion-play').click();page.wait_for_timeout(180)
            assert timer.input_value()!=before
            page.locator('#paired-motion-play').click()
            page.locator('#paired-curve-metric').select_option('jerk_component_rms_m_s3')
            checks.append({'method':method,'diagrams':2,'synchronized_playback':True,'jerk_curve_control':True})
        else:checks.append({'method':method,'diagrams':2,'synchronized_playback':False})
    for method in ['MotionGPT','PhysDiff','UHC','PHC_MJX','SMPLSim']:
        page.get_by_label('候选方法图',exact=True).select_option(method)
        for kind in ['backbone','loop']:
            svg=page.locator(f'svg[aria-label="candidate-{method.lower()} {kind}"]')
            assert svg.count()==1
            svg.screenshot(path=str(OUT/f'candidate_{method}_{kind}.png'))
        checks.append({'candidate':method,'diagrams':2})
    section=page.locator('#supplement-evidence')
    section.get_by_role('button',name='左右姿态与评分范围',exact=True).click()
    page.get_by_label('补实验方法',exact=True).select_option('motiongpt')
    page.get_by_label('评分曲线规则',exact=True).select_option('right_REBA')
    curve=page.get_by_role('img',name='左右人体工学评分范围时间曲线')
    curve.screenshot(path=str(OUT/'motiongpt_posture_curve.png'))
    assert curve.locator('polygon').count()==1
    for method in ['uhc','phc_tracker']:
        page.get_by_label('补实验方法',exact=True).select_option(method)
        assert page.get_by_role('img',name='左右人体工学评分范围时间曲线').locator('polygon').count()==1
        checks.append({'posture_method':method,'score_range_curve':True})
    section.get_by_role('button',name='共同参考跟踪',exact=True).click()
    selector=page.get_by_label('共同跟踪参考动作',exact=True)
    sources=selector.locator('option').evaluate_all('(options)=>options.map(o=>o.value)')
    assert len(sources)==8
    for source in sources:
        selector.select_option(source)
        assert page.locator('svg[aria-label$="共同参考跟踪骨架"]').count()==3
    timer=page.get_by_label('共同跟踪时间',exact=True)
    before=timer.input_value();page.get_by_label('共同跟踪同步播放',exact=True).click();page.wait_for_timeout(250)
    assert timer.input_value()!=before
    page.get_by_label('共同跟踪同步播放',exact=True).click()
    page.get_by_label('共同跟踪视野',exact=True).select_option('10')
    page.locator('[data-component-id="supplement-tracking-player"]').screenshot(path=str(OUT/'shared_tracking_player.png')) if page.locator('[data-component-id="supplement-tracking-player"]').count() else page.locator('svg[aria-label$="共同参考跟踪骨架"]').first.screenshot(path=str(OUT/'shared_tracking_player.png'))
    checks.append({'common_tracking_sources':8,'conditions_per_source':3,'playback_advances':True})
    section.get_by_role('button',name='同动作收益与代价',exact=True).click()
    page.get_by_label('联合评价方法',exact=True).select_option('kimodo')
    assert page.get_by_label('联合评价时钟',exact=True).input_value()=='historical'
    assert page.get_by_label('联合评价配对',exact=True).locator('option').count()==416
    checks.append({'kimodo_historical_joint_pairs':416})
    section.get_by_role('button',name='阶段结论与扩样',exact=True).click()
    assert section.get_by_text('没有启动扩样',exact=False).count()>0
    checks.append({'phase1_capacity_view':True})
    section.get_by_role('button',name='真实在线更新',exact=True).click()
    page.get_by_label('补实验方法',exact=True).select_option('dart')
    goal=page.get_by_role('img',name='在线事件后的目标距离曲线')
    goal.screenshot(path=str(OUT/'online_goal_curve.png'))
    # Historical report remains an interactive iframe, not flattened text.
    page.locator('#historical-report-select').select_option('20261007_abeh_baseline_interactive_report.html')
    frame=page.frame_locator('#historical-html-frame')
    frame.locator('.play').first.wait_for(timeout=15000)
    count=frame.locator('.play').count()
    frame.locator('.play').first.click()
    video=frame.locator('video').first
    video.wait_for(state='attached')
    page.wait_for_timeout(1000)
    media=video.evaluate('(v)=>({readyState:v.readyState,currentTime:v.currentTime,error:v.error?.message||null,source:v.currentSrc.slice(0,30)})')
    assert media['readyState']>=2 and not media['error'],media
    checks.append({'historical_1007_video_buttons':count,'media':media})
    assert not errors,errors
    (OUT/'full_interaction_check.json').write_text(json.dumps({'page_errors':errors,'checks':checks,'scope':'Actual browser interaction checks; scientific validity and statistical coverage are separate.'},indent=2))
    browser.close()
print(json.dumps({'checks':len(checks),'page_errors':errors}))
