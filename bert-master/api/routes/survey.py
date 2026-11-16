"""问卷管理 CRUD + 答卷提交/查询"""

import json
import math
import logging
import datetime as _dt
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..utils.db import mysql_conn
from ..utils.auth import verify_token, get_current_user
from ..utils.rbac import (
    normalize_role,
    is_any_admin,
    is_super_admin,
    is_school_admin,
    questionnaire_manage_allowed,
)
from ..utils.admin_permissions import PERM_SURVEY_MANAGE, load_user_permissions

router = APIRouter(prefix="/survey", tags=["survey"])
logger = logging.getLogger(__name__)

_SEEDED = False


# --------------- Pydantic 模型 ---------------

class QuestionnaireCreate(BaseModel):
    title: str
    description: str = ""
    scoring_rules: dict = {}
    score_multiplier: float = 1.0
    items: list = []
    # 仅超级管理员可指定归属校；校级管理员强制为本校
    school_id: Optional[str] = None


class SubmitAnswer(BaseModel):
    questionnaire_id: int
    answers: dict
    token: Optional[str] = None


# --------------- 预置问卷数据 ---------------

_PRESET_QUESTIONNAIRES: List[dict] = []


def _opt4():
    return [
        {"value": 0, "label": "完全没有"},
        {"value": 1, "label": "有几天"},
        {"value": 2, "label": "一半以上天数"},
        {"value": 3, "label": "几乎每天"},
    ]


def _build_presets():
    global _PRESET_QUESTIONNAIRES
    if _PRESET_QUESTIONNAIRES:
        return

    # ---- PHQ-9 ----
    phq9_questions = [
        "做事情提不起兴趣，或觉得做什么都没意思",
        "感到心情低落、沮丧或绝望",
        "入睡困难、容易醒来，或睡得太多",
        "感到疲倦或精力不足",
        "食欲不振或吃得过多",
        "对自己感到不好，或者觉得自己很失败，或让家人/自己失望",
        "做事时难以集中注意力，例如看书或看手机时容易走神",
        "行动或说话比平时慢很多，或者相反——坐立不安、动来动去",
        "有过觉得活着太痛苦、不如不活了，或产生伤害自己的念头",
    ]
    _PRESET_QUESTIONNAIRES.append({
        "title": "PHQ-9 抑郁症筛查量表",
        "description": "参考国际通用 PHQ-9，评估过去两周内的抑郁症状。9题，每题0-3分，总分0-27。",
        "score_multiplier": 1.0,
        "scoring_rules": {
            "ranges": [
                {"min": 0, "max": 4, "level": "无或最轻微抑郁", "suggestion": "当前整体状态相对稳定，可继续保持良好的作息与社交习惯。"},
                {"min": 5, "max": 9, "level": "轻度抑郁可能", "suggestion": "最近情绪可能有一定波动，建议关注作息与压力，必要时与信任的人聊一聊。"},
                {"min": 10, "max": 14, "level": "中度抑郁可能", "suggestion": "建议尽快和学校心理中心或专业机构取得联系，获得更系统的评估与支持。"},
                {"min": 15, "max": 19, "level": "中重度抑郁可能", "suggestion": "近期情绪困扰较明显，强烈建议尽快预约专业心理咨询或精神专科门诊。"},
                {"min": 20, "max": 27, "level": "重度抑郁可能", "suggestion": "请高度重视当前状态，尤其若伴随自伤念头，请立即联系家人、老师或急救热线。"},
            ]
        },
        "items": [{"question_text": q, "options": _opt4(), "is_reverse": False} for q in phq9_questions],
    })

    # ---- GAD-7 ----
    gad7_questions = [
        "感到紧张、焦虑或急切",
        "不能停止或控制担忧",
        "对各种各样的事情担忧过多",
        "很难放松下来",
        "由于不安而无法静坐",
        "变得容易烦恼或急躁",
        "感到似乎将有可怕的事情发生",
    ]
    _PRESET_QUESTIONNAIRES.append({
        "title": "GAD-7 焦虑症筛查量表",
        "description": "参考国际通用 GAD-7，评估过去两周内的焦虑症状。7题，每题0-3分，总分0-21。",
        "score_multiplier": 1.0,
        "scoring_rules": {
            "ranges": [
                {"min": 0, "max": 4, "level": "无或最轻微焦虑", "suggestion": "当前焦虑水平较低，保持规律作息和适度运动即可。"},
                {"min": 5, "max": 9, "level": "轻度焦虑", "suggestion": "可尝试深呼吸、冥想等放松技巧，注意减压。"},
                {"min": 10, "max": 13, "level": "中度焦虑", "suggestion": "建议与心理咨询师沟通，了解焦虑的来源并学习应对策略。"},
                {"min": 14, "max": 18, "level": "中重度焦虑", "suggestion": "强烈建议寻求专业心理或精神科帮助，必要时考虑药物辅助。"},
                {"min": 19, "max": 21, "level": "重度焦虑", "suggestion": "请尽快就医，严重焦虑可能严重影响日常功能，需要专业干预。"},
            ]
        },
        "items": [{"question_text": q, "options": _opt4(), "is_reverse": False} for q in gad7_questions],
    })

    # ---- SDS (Zung) ----
    sds_questions = [
        "我觉得闷闷不乐，情绪低沉",
        "我觉得一天之中早晨最好",
        "我一阵阵哭出来或觉得想哭",
        "我晚上睡眠不好",
        "我吃得跟平常一样多",
        "我与异性密切接触时和以往一样感到愉快",
        "我发觉我的体重在下降",
        "我有便秘的苦恼",
        "我心跳比平常快",
        "我无缘无故地感到疲乏",
        "我的头脑跟平常一样清楚",
        "我觉得经常做的事情并没有困难",
        "我觉得不安而平静不下来",
        "我对将来抱有希望",
        "我比平常容易生气激动",
        "我觉得做出决定是容易的",
        "我觉得自己是个有用的人，有人需要我",
        "我的生活过得很有意思",
        "我认为如果我死了别人会生活得好些",
        "平常感兴趣的事我仍然照样感兴趣",
    ]
    sds_opt = [
        {"value": 1, "label": "没有或很少时间"},
        {"value": 2, "label": "小部分时间"},
        {"value": 3, "label": "相当多时间"},
        {"value": 4, "label": "绝大部分或全部时间"},
    ]
    sds_reverse_indices = {1, 4, 5, 10, 11, 13, 15, 16, 17, 19}
    _PRESET_QUESTIONNAIRES.append({
        "title": "SDS 抑郁自评量表",
        "description": "Zung 抑郁自评量表，20题。粗分×1.25取整为标准分。含反向计分题。",
        "score_multiplier": 1.25,
        "scoring_rules": {
            "ranges": [
                {"min": 0, "max": 52, "level": "正常", "suggestion": "目前情绪状态在正常范围内。"},
                {"min": 53, "max": 62, "level": "轻度抑郁", "suggestion": "存在轻度抑郁倾向，建议适当调节并关注情绪变化。"},
                {"min": 63, "max": 72, "level": "中度抑郁", "suggestion": "建议寻求心理咨询帮助，进行更详细的评估。"},
                {"min": 73, "max": 100, "level": "重度抑郁", "suggestion": "请尽快就医，寻求专业精神科帮助。"},
            ]
        },
        "items": [
            {"question_text": q, "options": sds_opt, "is_reverse": (i in sds_reverse_indices)}
            for i, q in enumerate(sds_questions)
        ],
    })

    # ---- SAS (Zung) ----
    sas_questions = [
        "我觉得比平常容易紧张和着急",
        "我无缘无故地感到害怕",
        "我容易心里烦乱或觉得惊恐",
        "我觉得我可能将要发疯",
        "我觉得一切都很好，也不会发生什么不幸",
        "我手脚发抖打颤",
        "我因为头痛、头颈痛和背痛而苦恼",
        "我感觉容易衰弱和疲乏",
        "我觉得心平气和，并且容易安静坐着",
        "我觉得心跳得很快",
        "我因为一阵阵头晕而苦恼",
        "我有晕倒发作或觉得要晕倒似的",
        "我呼气吸气都感到很容易",
        "我手脚麻木和刺痛",
        "我因为胃痛和消化不良而苦恼",
        "我常常要小便",
        "我的手常常是干燥温暖的",
        "我脸红发热",
        "我容易入睡并且一夜睡得很好",
        "我做噩梦",
    ]
    sas_opt = sds_opt[:]
    sas_reverse_indices = {4, 8, 12, 16, 18}
    _PRESET_QUESTIONNAIRES.append({
        "title": "SAS 焦虑自评量表",
        "description": "Zung 焦虑自评量表，20题。粗分×1.25取整为标准分。含反向计分题。",
        "score_multiplier": 1.25,
        "scoring_rules": {
            "ranges": [
                {"min": 0, "max": 49, "level": "正常", "suggestion": "焦虑水平在正常范围内。"},
                {"min": 50, "max": 59, "level": "轻度焦虑", "suggestion": "存在轻度焦虑倾向，可尝试放松训练来缓解。"},
                {"min": 60, "max": 69, "level": "中度焦虑", "suggestion": "建议寻求心理咨询帮助，了解焦虑源并学习应对方法。"},
                {"min": 70, "max": 100, "level": "重度焦虑", "suggestion": "请尽快就医，严重焦虑需要专业干预。"},
            ]
        },
        "items": [
            {"question_text": q, "options": sas_opt, "is_reverse": (i in sas_reverse_indices)}
            for i, q in enumerate(sas_questions)
        ],
    })


def _seed_presets():
    """首次启动时将预置问卷写入数据库（幂等）。"""
    global _SEEDED
    if _SEEDED:
        return
    _build_presets()
    with mysql_conn() as conn:
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM questionnaires")
                row = cur.fetchone()
                if row and row.get("cnt", 0) > 0:
                    _SEEDED = True
                    return
                for preset in _PRESET_QUESTIONNAIRES:
                    cur.execute(
                        """INSERT INTO questionnaires
                           (title, description, scoring_rules, score_multiplier, is_active, school_id, created_at)
                           VALUES (%s,%s,%s,%s,TRUE,'',%s)""",
                        (preset["title"], preset["description"],
                         json.dumps(preset["scoring_rules"], ensure_ascii=False),
                         preset["score_multiplier"], _dt.datetime.utcnow()),
                    )
                    qid = cur.lastrowid
                    for idx, item in enumerate(preset["items"]):
                        cur.execute(
                            """INSERT INTO questionnaire_items
                               (questionnaire_id, item_index, question_text, options, is_reverse)
                               VALUES (%s,%s,%s,%s,%s)""",
                            (qid, idx, item["question_text"],
                             json.dumps(item["options"], ensure_ascii=False),
                             item.get("is_reverse", False)),
                        )
            conn.commit()
            _SEEDED = True
            logger.info("预置问卷数据写入完成")
        except Exception as e:
            logger.warning(f"预置问卷写入失败: {e}")


# --------------- API 路由 ---------------

def _can_student_see_questionnaire(school_id_on_q: str, student_school: str) -> bool:
    qs = (school_id_on_q or "").strip()
    if qs == "":
        return True
    return qs == (student_school or "").strip()


@router.get("/list")
def list_questionnaires(request: Request):
    """已启用问卷：匿名仅公共模板；登录学生可见公共+本校。"""
    _seed_presets()
    payload = get_current_user(request)
    student_school = ""
    if payload and normalize_role(payload.get("role")) == "student":
        student_school = (payload.get("school") or "").strip()
    with mysql_conn() as conn:
        if conn is None:
            return {"items": []}
        with conn.cursor() as cur:
            if student_school:
                cur.execute(
                    """SELECT q.id, q.title, q.description, q.score_multiplier, q.is_active,
                              (SELECT COUNT(*) FROM questionnaire_items WHERE questionnaire_id=q.id) AS item_count
                       FROM questionnaires q
                       WHERE q.is_active=TRUE
                         AND (COALESCE(q.school_id,'')='' OR q.school_id=%s)
                       ORDER BY q.id""",
                    (student_school,),
                )
            else:
                cur.execute(
                    """SELECT q.id, q.title, q.description, q.score_multiplier, q.is_active,
                              (SELECT COUNT(*) FROM questionnaire_items WHERE questionnaire_id=q.id) AS item_count
                       FROM questionnaires q
                       WHERE q.is_active=TRUE AND COALESCE(q.school_id,'')=''
                       ORDER BY q.id""",
                )
            return {"items": cur.fetchall() or []}


@router.get("/all")
def list_all_questionnaires(request: Request):
    """管理员：本校 + 公共模板；超级管理员：全部。"""
    user = get_current_user(request)
    if not user or not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    _seed_presets()
    vr = normalize_role(user.get("role"))
    my_school = (user.get("school") or "").strip()
    with mysql_conn() as conn:
        if conn is None:
            return {"items": []}
        with conn.cursor() as cur:
            if is_super_admin(vr):
                cur.execute(
                    """SELECT q.id, q.title, q.description, q.score_multiplier, q.is_active, q.school_id,
                              (SELECT COUNT(*) FROM questionnaire_items WHERE questionnaire_id=q.id) AS item_count
                       FROM questionnaires q ORDER BY q.id""",
                )
            else:
                cur.execute(
                    """SELECT q.id, q.title, q.description, q.score_multiplier, q.is_active, q.school_id,
                              (SELECT COUNT(*) FROM questionnaire_items WHERE questionnaire_id=q.id) AS item_count
                       FROM questionnaires q
                       WHERE COALESCE(q.school_id,'')='' OR q.school_id=%s
                       ORDER BY q.id""",
                    (my_school,),
                )
            return {"items": cur.fetchall() or []}


@router.get("/{qid}")
def get_questionnaire(qid: int, request: Request):
    """获取问卷详情；学生仅能访问公共或本校问卷。"""
    _seed_presets()
    payload = get_current_user(request)
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM questionnaires WHERE id=%s", (qid,))
            q = cur.fetchone()
            if not q:
                raise HTTPException(404, "问卷不存在")
            qs = (q.get("school_id") or "").strip()
            if payload:
                pr = normalize_role(payload.get("role"))
                if pr == "student":
                    ss = (payload.get("school") or "").strip()
                    if not _can_student_see_questionnaire(qs, ss):
                        raise HTTPException(404, "问卷不存在")
                elif is_any_admin(pr):
                    if not is_super_admin(pr):
                        my = (payload.get("school") or "").strip()
                        if qs != "" and qs != my:
                            raise HTTPException(404, "问卷不存在")
            else:
                if qs != "":
                    raise HTTPException(404, "问卷不存在")
            cur.execute(
                "SELECT * FROM questionnaire_items WHERE questionnaire_id=%s ORDER BY item_index",
                (qid,))
            items = cur.fetchall() or []
            for it in items:
                if isinstance(it.get("options"), str):
                    it["options"] = json.loads(it["options"])
            if isinstance(q.get("scoring_rules"), str):
                q["scoring_rules"] = json.loads(q["scoring_rules"])
            q["items"] = items
            return q


@router.post("/{qid}/submit")
def submit_answer(qid: int, req: SubmitAnswer):
    """提交答卷，后端计算得分并存储。"""
    _seed_presets()
    user_id = None
    payload = None
    if req.token:
        payload = verify_token(req.token)
        if payload:
            user_id = payload.get("uid")

    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM questionnaires WHERE id=%s", (qid,))
            q = cur.fetchone()
            if not q:
                raise HTTPException(404, "问卷不存在")
            qs = (q.get("school_id") or "").strip()
            if payload:
                pr = normalize_role(payload.get("role"))
                if pr == "student":
                    if not _can_student_see_questionnaire(qs, (payload.get("school") or "").strip()):
                        raise HTTPException(404, "问卷不存在")
                elif is_any_admin(pr):
                    if not is_super_admin(pr):
                        my = (payload.get("school") or "").strip()
                        if qs != "" and qs != my:
                            raise HTTPException(404, "问卷不存在")
            else:
                if qs != "":
                    raise HTTPException(404, "问卷不存在")
            cur.execute(
                "SELECT * FROM questionnaire_items WHERE questionnaire_id=%s ORDER BY item_index",
                (qid,))
            items = cur.fetchall() or []

        multiplier = float(q.get("score_multiplier") or 1.0)
        scoring_rules = q.get("scoring_rules")
        if isinstance(scoring_rules, str):
            scoring_rules = json.loads(scoring_rules)

        raw_score = 0
        for it in items:
            idx_key = str(it["item_index"])
            ans_val = req.answers.get(idx_key)
            if ans_val is None:
                continue
            ans_val = int(ans_val)
            if it.get("is_reverse"):
                opts = it.get("options")
                if isinstance(opts, str):
                    opts = json.loads(opts)
                max_val = max(o["value"] for o in opts)
                min_val = min(o["value"] for o in opts)
                ans_val = max_val + min_val - ans_val
            raw_score += ans_val

        standard_score = raw_score
        if multiplier != 1.0:
            standard_score = math.floor(raw_score * multiplier)

        result_level = ""
        suggestion = ""
        if scoring_rules and "ranges" in scoring_rules:
            check_score = standard_score if multiplier != 1.0 else raw_score
            for r in scoring_rules["ranges"]:
                if r["min"] <= check_score <= r["max"]:
                    result_level = r.get("level", "")
                    suggestion = r.get("suggestion", "")
                    break

        result_school = ""
        with conn.cursor() as cur:
            if user_id:
                cur.execute("SELECT school_id FROM users WHERE id=%s", (user_id,))
                urow = cur.fetchone()
                if urow:
                    result_school = (urow.get("school_id") or "").strip()
            cur.execute(
                """INSERT INTO questionnaire_results
                   (user_id, questionnaire_id, answers, raw_score, standard_score, result_level, school_id, created_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (user_id, qid, json.dumps(req.answers, ensure_ascii=False),
                 raw_score, standard_score, result_level, result_school, _dt.datetime.utcnow()),
            )
        conn.commit()

    return {
        "raw_score": raw_score,
        "standard_score": standard_score,
        "result_level": result_level,
        "suggestion": suggestion,
        "multiplier": multiplier,
    }


@router.post("/create")
def create_questionnaire(req: QuestionnaireCreate, request: Request):
    """管理员创建新问卷（校级归属本校；超管可指定 school_id 或空=公共）。"""
    user = get_current_user(request)
    if not user or not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    vr = normalize_role(user.get("role"))
    perms = load_user_permissions(user.get("uid"))
    if not is_super_admin(vr) and PERM_SURVEY_MANAGE not in perms:
        raise HTTPException(403, "当前账号暂无问卷编辑权限")
    my_school = (user.get("school") or "").strip()
    if is_super_admin(vr):
        q_school = (req.school_id if req.school_id is not None else "").strip()
    else:
        if not my_school:
            raise HTTPException(400, "校级管理员须填写学校标识 school_id")
        q_school = my_school
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO questionnaires
                   (title, description, scoring_rules, score_multiplier, is_active, school_id, created_by, created_at)
                   VALUES (%s,%s,%s,%s,TRUE,%s,%s,%s)""",
                (req.title, req.description,
                 json.dumps(req.scoring_rules, ensure_ascii=False),
                 req.score_multiplier, q_school, user.get("uid"),
                 _dt.datetime.utcnow()),
            )
            qid = cur.lastrowid
            for idx, item in enumerate(req.items):
                cur.execute(
                    """INSERT INTO questionnaire_items
                       (questionnaire_id, item_index, question_text, options, is_reverse)
                       VALUES (%s,%s,%s,%s,%s)""",
                    (qid, idx, item.get("question_text", ""),
                     json.dumps(item.get("options", []), ensure_ascii=False),
                     item.get("is_reverse", False)),
                )
        conn.commit()
    return {"id": qid, "message": "问卷创建成功"}


@router.put("/{qid}/toggle")
def toggle_questionnaire(qid: int, request: Request):
    """管理员启用/停用问卷。"""
    user = get_current_user(request)
    if not user or not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    perms = load_user_permissions(user.get("uid"))
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute("SELECT is_active, school_id FROM questionnaires WHERE id=%s", (qid,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "问卷不存在")
            qs = row.get("school_id")
            if not questionnaire_manage_allowed(
                user.get("role"), (user.get("school") or "").strip(), qs, perms,
            ):
                raise HTTPException(403, "无权修改该问卷（公共模板仅超级管理员可管理）")
            new_state = not row["is_active"]
            cur.execute("UPDATE questionnaires SET is_active=%s WHERE id=%s", (new_state, qid))
        conn.commit()
    return {"id": qid, "is_active": new_state}


@router.put("/{qid}")
def update_questionnaire(qid: int, req: QuestionnaireCreate, request: Request):
    """管理员更新问卷（覆盖标题、说明、计分规则与全部题目）。"""
    user = get_current_user(request)
    if not user or not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    perms = load_user_permissions(user.get("uid"))
    if not req.title.strip():
        raise HTTPException(400, "标题不能为空")
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, school_id FROM questionnaires WHERE id=%s", (qid,))
                qrow = cur.fetchone()
                if not qrow:
                    raise HTTPException(404, "问卷不存在")
                if not questionnaire_manage_allowed(
                    user.get("role"), (user.get("school") or "").strip(), qrow.get("school_id"), perms,
                ):
                    raise HTTPException(403, "无权修改该问卷")
                cur.execute(
                    """UPDATE questionnaires SET title=%s, description=%s,
                       scoring_rules=%s, score_multiplier=%s WHERE id=%s""",
                    (
                        req.title.strip(),
                        req.description or "",
                        json.dumps(req.scoring_rules or {}, ensure_ascii=False),
                        float(req.score_multiplier or 1.0),
                        qid,
                    ),
                )
                cur.execute(
                    "DELETE FROM questionnaire_items WHERE questionnaire_id=%s", (qid,)
                )
                for idx, item in enumerate(req.items or []):
                    cur.execute(
                        """INSERT INTO questionnaire_items
                           (questionnaire_id, item_index, question_text, options, is_reverse)
                           VALUES (%s,%s,%s,%s,%s)""",
                        (
                            qid,
                            idx,
                            item.get("question_text", ""),
                            json.dumps(item.get("options", []), ensure_ascii=False),
                            bool(item.get("is_reverse", False)),
                        ),
                    )
            conn.commit()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"更新失败: {e}")
    return {"id": qid, "message": "问卷已更新"}


@router.delete("/{qid}")
def delete_questionnaire(qid: int, request: Request):
    """管理员删除问卷（软删除 = 停用）。"""
    user = get_current_user(request)
    if not user or not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    perms = load_user_permissions(user.get("uid"))
    with mysql_conn() as conn:
        if conn is None:
            raise HTTPException(503, "数据库不可用")
        with conn.cursor() as cur:
            cur.execute("SELECT school_id FROM questionnaires WHERE id=%s", (qid,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "问卷不存在")
            if not questionnaire_manage_allowed(
                user.get("role"), (user.get("school") or "").strip(), row.get("school_id"), perms,
            ):
                raise HTTPException(403, "无权操作该问卷")
            cur.execute("UPDATE questionnaires SET is_active=FALSE WHERE id=%s", (qid,))
        conn.commit()
    return {"message": "问卷已停用"}


@router.get("/results/list")
def list_results(request: Request, questionnaire_id: Optional[int] = None, limit: int = 100):
    """管理员查询答卷：本校+公共池；超级管理员全量。"""
    user = get_current_user(request)
    if not user or not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    vr = normalize_role(user.get("role"))
    my_school = (user.get("school") or "").strip()
    with mysql_conn() as conn:
        if conn is None:
            return {"items": []}
        with conn.cursor() as cur:
            base = """SELECT r.*, q.title AS questionnaire_title, u.username, u.display_name
                       FROM questionnaire_results r
                       LEFT JOIN questionnaires q ON r.questionnaire_id=q.id
                       LEFT JOIN users u ON r.user_id=u.id
                       WHERE """
            if is_super_admin(vr):
                if questionnaire_id:
                    cur.execute(
                        base + "r.questionnaire_id=%s ORDER BY r.created_at DESC LIMIT %s",
                        (questionnaire_id, limit),
                    )
                else:
                    cur.execute(
                        base + "1=1 ORDER BY r.created_at DESC LIMIT %s",
                        (limit,),
                    )
            else:
                tenant = "r.school_id=%s"
                if questionnaire_id:
                    cur.execute(
                        base + tenant + " AND r.questionnaire_id=%s ORDER BY r.created_at DESC LIMIT %s",
                        (my_school, questionnaire_id, limit),
                    )
                else:
                    cur.execute(
                        base + tenant + " ORDER BY r.created_at DESC LIMIT %s",
                        (my_school, limit),
                    )
            return {"items": cur.fetchall() or []}
