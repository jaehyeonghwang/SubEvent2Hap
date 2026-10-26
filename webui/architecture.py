"""Live framework description for the /architecture page.

Everything here is read from the loaded modules or the files on disk, so the page tracks
the code without hand-maintained copies. The diagrams in architecture.html and the change
log (architecture_log.json) are the only hand-written parts.
"""
import json
import pathlib
import re
import time
from dataclasses import asdict

import bounce as bp
import compose as cp
import decompose as dc
import model_planner as mp
import physical_models as pm

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = pathlib.Path(__file__).with_name("architecture_log.json")
STARTED = time.time()

# Korean display text for the page only. The English `applicability` is what the LM reads and
# stays the source of truth; a model/parameter missing here is shown in English and flagged.
MODEL_KO = {
    "vertical_bounce": ("수직 바운스", "정지 상태에서 떨어진 물체가 고정된 평평한 바닥에서 한 번 이상 튀는 과정. 회전 없는 점 물체로 근사. 드리블처럼 힘을 가하는 동작이나 이후 굴러가는 동작은 포함하지 않음."),
    "free_fall": ("자유 낙하", "바닥에 처음 닿을 때까지의 수직 비행. 이 모델은 진동을 만들지 않으며, 이어지는 충돌은 별도 단계로 분리."),
    "single_impact": ("단일 충돌", "고정된 면에 수직으로 한 번 부딪히는 충돌. 반복 동작이나 별도로 언급된 울림 꼬리는 포함하지 않음."),
    "sliding": ("미끄러짐", "평평한 수평면에서의 미끄러짐·끌림 (쿨롱 마찰). 힘 없이 미끄러지다 멈추거나, 외부 힘으로 일정 속도 유지. 굴러감·곡면·경사면·변하는 힘은 제외."),
    "rolling": ("구름", "평평한 수평면에서 일정한 구름 저항으로 감속하며 굴러감. 곡면·경사면은 미지원. 바운스에서 구름으로의 전환은 추론하지 않음."),
    "damped_oscillation": ("감쇠 진동", "자극 뒤에 점점 잦아드는 울림 하나. 자극이 된 충돌 자체는 필요하면 별도 단계로."),
    "periodic": ("주기 진동", "모터·공구처럼 일정하게 계속되는 주기 진동. 사인파로 합성. 몇 번의 두드림 같은 이산 동작은 해당 없음."),
    "stochastic": ("무작위 접촉", "빗방울·자갈처럼 불규칙하고 서로 독립적인 접촉. 포아송 과정 또는 횟수가 정해진 조건부 형태로 표현."),
    "rhythmic_action": ("리듬 동작", "노크·걸음·두드림·드리블처럼 의도적으로 반복하는 동작. 박자는 행동 설계이며 물리적 반발이 아님."),
}
FAMILY_KO = {"collision": "충돌", "flight": "비행", "friction": "마찰", "damping": "감쇠", "periodic": "주기",
             "stochastic": "확률", "behavior": "행동"}
SCOPE_KO = {"scenario": "시나리오", "basic": "기본", "process": "과정"}
PARAM_KO = {"height_m": "높이 (m)", "vertical_velocity_m_s": "수직 속도 (m/s)", "normal_speed_m_s": "충돌 속도 (m/s)",
            "speed_m_s": "수평 속도 (m/s)", "mass_kg": "질량 (kg)", "restitution": "반발계수", "start_s": "시작 시각 (s)",
            "duration_s": "지속 시간 (s)", "count": "횟수", "friction_coefficient": "마찰계수",
            "rolling_resistance": "구름 저항", "frequency_hz": "진동수 (Hz)", "rpm": "회전수 (rpm)",
            "damping_ratio": "감쇠비", "excitation_energy_J": "자극 에너지 (J)", "rate_hz": "발생률 (회/s)",
            "seed": "난수 seed", "interval_s": "간격 (s)", "tempo_bpm": "템포 (bpm)",
            "interval_ratio": "간격 비율", "strength_ratio": "세기 비율"}

# Who owns each planner stage (for colouring); derived names are matched by prefix.
STAGE_OWNER = {"model_selection": "LM", "setup_": "LM", "estimate_": "LM", "structure_review": "LM",
               "structure_split": "LM", "split_review": "LM", "state_binding": "LM", "atomic_fallback": "LM"}


def _planner_stages():
    """Stage names in the order they appear in model_planner.decompose (parsed from source)."""
    src = pathlib.Path(mp.__file__).read_text(encoding="utf-8")
    found = []
    for m in re.finditer(r"run\(\s*'([a-z_]+)'(\s*\+\s*selected)?", src):
        name = m.group(1) + ("<model>" if m.group(2) else "")
        if name not in found:
            found.append(name)
    return found


def _staged_stages():
    """LM stage names of decompose.decompose in source order."""
    src = pathlib.Path(dc.__file__).read_text(encoding="utf-8")
    body = src[src.index("def decompose("):]
    return list(dict.fromkeys(re.findall(r'(?:run|_stage)\(\s*"([a-z_]+)"', body)))


ABLATION_LABELS = {
    "full": "모델 우선 계층 플래너 (전체)", "-models": "계층 플래너 − 물리 모델",
    "-structure_review": "계층 플래너 − 구조 검토", "-split_review": "계층 플래너 − 분해 검토",
    "-state_binding": "계층 플래너 − 상태 연결", "-alternative_selection": "계층 플래너 − 대체 모델 재선택",
    "+estimate": "계층 플래너 + LM 물리량 추정", "staged": "LM 사건 분해 (기본)",
    "staged-review": "LM 사건 분해 − 검토",
    "staged@gpt-4o": "LM 사건 분해 · GPT-4o",
}


def _ablation():
    """Latest planner ablation summary (scripts/ablation.py score), if it exists."""
    path = ROOT / "out" / "ablation" / "scores_v2.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    eval_meta = json.loads((ROOT / "data" / "eval_set_v2.json").read_text(encoding="utf-8"))
    keep = ("variant", "n", "renderable", "items_all_pass", "count", "order", "lm_calls")
    return {"eval_version": eval_meta["version"], "status": eval_meta["status"], "scored": path.stat().st_mtime,
            "rows": [{**{k: r.get(k) for k in keep}, "label": ABLATION_LABELS.get(r["variant"], r["variant"])}
                     for r in rows]}


def _files():
    rows = []
    for folder, pattern in (("scripts", "*.py"), ("scripts", "*.md"), ("webui", "*.py"), ("webui", "*.html")):
        for p in sorted((ROOT / folder).glob(pattern)):
            st = p.stat()
            rows.append({"path": f"{folder}/{p.name}", "lines": sum(1 for _ in p.open(encoding="utf-8", errors="ignore")),
                         "modified": st.st_mtime, "after_server_start": st.st_mtime > STARTED and p.suffix == ".py"})
    return rows


def _routes(app):
    out = []
    for r in app.routes:
        methods = sorted(getattr(r, "methods", None) or [])
        if r.path.startswith(("/openapi", "/docs", "/redoc")):
            continue
        out.append({"methods": [m for m in methods if m != "HEAD"], "path": r.path,
                    "doc": (getattr(r, "endpoint", None).__doc__ or "").strip().split("\n")[0]})
    return out


def describe(app, status):
    catalog = []
    for m in pm.MODELS:
        d = asdict(m)
        ko = MODEL_KO.get(m.name)
        d.update(name_ko=ko[0] if ko else None, applicability_ko=ko[1] if ko else None,
                 family_ko=FAMILY_KO.get(m.family, m.family), scope_ko=SCOPE_KO.get(m.scope, m.scope))
        d["parameters"] = [{"name": k, "name_ko": PARAM_KO.get(k), "description": pm.FIELDS[k][2],
                            "estimable": k in pm.ESTIMABLE, "inherits_from": pm.INHERITS.get(k)}
                           for k in m.parameters]
        catalog.append(d)
    try:
        log = json.loads(LOG.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log = []
    return {
        "generated": time.time(),
        "server_started": STARTED,
        "versions": {"model_planner": mp.VERSION, "staged_decomposer": dc.PROMPT_VERSION, "ball_physics": bp.VERSION},
        "planners": [
            {"id": "staged", "module": "decompose.py", "version": dc.PROMPT_VERSION, "default": True,
             "summary": "기본. LM 4단계(사건 → 검토 → 타이밍 → 캡션)로 sub-event를 만듭니다."},
            {"id": "model_first", "module": "model_planner.py", "version": mp.VERSION, "default": False,
             "summary": "실험용. LM이 물리 모델을 고르거나 동작을 분해하고, 코드가 선택된 모델을 계산합니다. 비교 실험에서 기본 경로보다 낮아 기본에서 제외."},
            {"id": "ball_physics", "module": "bounce.py", "version": bp.VERSION, "default": False,
             "summary": "실험용. LM이 조건을 한 번 추출하고 수직 바운스 계산기를 실행합니다. UI에서 값을 직접 덮어쓸 수 있습니다."},
        ],
        "staged_stages": _staged_stages(),
        "ablation": _ablation(),
        "planner_stages": [{"name": s, "owner": next((o for k, o in STAGE_OWNER.items() if s.startswith(k)), "LM")}
                           for s in _planner_stages()],
        "limits": {"max_depth": 3, "max_nodes": 15, "render_events": pm.MAX_RENDER_EVENTS, "seconds": 10,
                   "bounce_stop_ratio": pm.BOUNCE_STOP_RATIO},
        "catalog": catalog,
        "precedence": ["text_extracted_by_lm", "inherited_state", "estimated_by_lm", "assumed_default"],
        "estimable": list(pm.ESTIMABLE),
        "composer": {"default_gamma": cp.DEFAULT_GAMMA, "impact_max_duration_s": cp.IMPACT_MAX_DUR,
                     "attack_window_s": cp.ATTACK_WIN, "crossfade_s": cp.XFADE, "cut_fade_in_s": cp.CUT_FADE_IN,
                     "fade_out_s": cp.FADE_OUT, "max_batch": cp.MAX_BATCH},
        "status": status,
        "routes": _routes(app),
        "files": _files(),
        "log": log,
    }
