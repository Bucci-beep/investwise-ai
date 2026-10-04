"""Chat-first UI. The engine owns financial meanings and recording gates."""
from __future__ import annotations

import importlib
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.conversation_factory import build_conversation_engine  # noqa: E402
from ai.conversation_engine import ConversationEngine, OPTIONAL_CONTEXT  # noqa: E402
import ai.conversation_engine as conversation_module  # noqa: E402
import ai.conversation_factory as factory_module  # noqa: E402
import ai.bounded_answers as bounded_rules  # noqa: E402
from ai.conversation_state import AnswerStatus  # noqa: E402
from ai.pre_model_controls import check_pre_model_controls  # noqa: E402

st.set_page_config(page_title="InvestWise | A careful conversation", page_icon="🌿",
                   layout="centered", initial_sidebar_state="collapsed")

# Consent requires an explicit acknowledgement, never a qualified financial reply.
YES = re.compile(r"^(yes|yeah|yep|correct|right|that's right|that is right|yes,? that's right|yes,? that is right|that looks right|all correct|confirm|yes,? all answers are accurate)[.!]*$", re.I)
FINAL_ACCURACY_YES = re.compile(r"^(?:yes[,;]?\s*)?(?:this|the)\s+(?:(?:exact|whole)\s+)?summary\s+is\s+(?:accurate|correct)[.!]*$", re.I)
NO = re.compile(r"^(no|nope|not quite|not right|that's not right|that is not right|it's a little (?:bit )?different|it is a little (?:bit )?different|no,? (?:it's|it is) a little (?:bit )?different|change(?: my answer)?)[.!]*$", re.I)
SAVE_YES = re.compile(r"^(yes|yes,? (?:please )?save(?: it| this version| my profile)?|save(?: it| this version| my profile)?|i consent to save)[.!]*$", re.I)
SAVE_NO = re.compile(r"^(no|no,? (?:do not|don't) save|(?:do not|don't) save(?: it)?|(?:no,? )?leave it unsaved)[.!]*$", re.I)

EXPLANATIONS = {
    "D1": "I’m asking what you expect to use this money for, such as a home, retirement, everyday costs or unexpected expenses. More than one purpose is fine. What would you use it for?",
    "D2": "This is optional background. Your age alone does not determine your time horizon or how much loss you could manage. You can give a range or leave this unanswered.",
    "D3": "This means the existing amount you want us to discuss, rather than your total wealth. An approximate range is enough; please say which currency you mean.",
    "D4": "I’m asking where the same money is now. Easy-access savings can usually be withdrawn; fixed-term savings may have restrictions. Investments and pension savings are separate choices. More than one place is fine.",
    "D5": "This means the sources that currently pay for your everyday costs, such as wages, pension payments or withdrawals from savings. It can be more than one source.",
    "D6": "Imagine you could not use the money we’re discussing for one year. Would other income or money cover all, some or none of your necessary living costs?",
    "D7": "This asks whether any required loan or credit repayments are manageable now. You do not need to give the amount of your debt.",
    "D8": "This is about investments you have personally held, rather than products you have only read about. Holding several types is fine; experience does not automatically mean you want more risk.",
    "D9": "Count investment purchases or sales you made or instructed during the last 12 months. Ordinary shopping and bank transfers do not count. A range is enough.",
    "D10": "This means new money you put into investments during the last 12 months, rather than their current value or changes in market prices. A range and currency are enough.",
    "D11": "Investments can fall below the original amount. Some or all of that original money could be lost permanently; recovery is not guaranteed. What is your understanding of that possibility?",
    "D12": "I’m asking about the earliest realistic need for any of this same money, including living costs or unexpected needs. This may be sooner than your ideal investment goal. What do you know about when you might need it?",
    "D13": "For example, a 20% loss would turn £1,000 into £800. I’m asking what that loss would change in practice: paying essential costs, your budget or other plans. This is separate from how it would feel.",
    "D14": "This is your preferred trade-off between possible growth and changes in value. Greater possible growth can involve larger falls and permanent loss. There is no preferred answer.",
    "D15": "For this question only, assume essential costs and known plans are still covered. I’m asking whether the example fall would feel unacceptable, worrying but acceptable, or comfortable. Recovery is not guaranteed.",
}

CONTEXT_LABELS = {
    "pot_reference": "money in scope",
    "currency": "currency",
    "purpose_detail": "purpose detail",
    "holding_detail": "holding detail",
    "income_detail": "living-cost source detail",
    "experience_detail": "experience detail",
    "access_timing": "access timing",
    "usable_other_backup": "other usable resources",
    "funding_when_need_precedes_access": "funding before access",
    "dependency_review": "dependency check",
    "essential_funding_review": "essential-cost funding check",
}

# Neutral explanations for words used by the approved questions. These explain
# the question's meaning and do not recommend any product or selected answer.
TERM_EXPLANATIONS = [
    (r"\b(?:permanent(?:ly)?|lost|loss)\b", {"D11"}, "Permanent loss means that you may never get back some or all of the original money. Recovery is not guaranteed."),
    (r"\betf\b", {"D8"}, "An ETF is a fund traded on an exchange. It holds investments such as shares or bonds; its value can fall and money can be lost."),
    (r"\b(?:outside|new|gain|rise|value)\b", {"D10"}, "New outside money means fresh money you added from outside your investments. A rise in value and money reused from selling investments do not count."),
    (r"\b(?:required|repayment|student.loan)\b", {"D7"}, "A required repayment is a payment you must make now. Owing a balance does not by itself mean a payment is currently due."),
    (r"\bpension\b", {"D5"}, "Pension payments are money you receive for spending. Money still held inside a pension is different from payments currently covering your living costs."),
    (r"\b(?:transaction|groceries|card)\b", {"D9"}, "Here, a transaction means an investment purchase or sale that you made or instructed. Shopping, card payments and ordinary bank transfers do not count."),
    (r"\b(?:earliest|ideal|need)\b", {"D12"}, "Earliest means the first point when you may realistically need any of this same money. That may be sooner than the date you would ideally like to keep investing until."),
    (r"\btrade.off\b", {"D14"}, "A trade-off means choosing what matters more to you: limiting changes in value, balancing that with possible growth, or accepting larger changes for greater possible growth. Losses can happen with any investment."),
    (r"\b(?:necessary|living costs|holidays)\b", {"D6"}, "Necessary costs include basic housing, food, bills and required repayments. Holidays are another plan. This question asks whether other usable resources cover necessary costs for the whole year."),
    (r"\b(?:covered|bills|example|earlier)\b", {"D15"}, "Only for this feelings question, assume essentials and known plans are still covered. That assumption does not replace your earlier answer about the real practical effect of a loss."),
]


def term_explanation(qid: str | None, text: str) -> str | None:
    for pattern, questions, explanation in TERM_EXPLANATIONS:
        if qid in questions and re.search(pattern, text, re.I):
            return explanation
    return None


def context_text(details: Any) -> str:
    """Display only context that the engine explicitly retained for this answer."""
    if not isinstance(details, dict):
        return ""
    parts = []
    for key, value in details.items():
        if key in CONTEXT_LABELS and isinstance(value, str) and value.strip():
            label = CONTEXT_LABELS[key]
            parts.append(f"{label}: {value.strip()}")
    return "; ".join(parts)


def add_message(role: str, text: str, **metadata: Any) -> None:
    st.session_state.iw_messages.append({"role": role, "text": text, **metadata})
    engine = st.session_state.get("iw_engine")
    if engine is not None:
        # Transient per-turn evidence includes the actual explanation and
        # playback shown, not just the earlier request for help. The profile
        # save path deliberately excludes this dialogue/audit history.
        engine.state.log("dialogue_message", metadata.get("qid") or engine.state.current_question_id,
                         role=role, text=text)


def refresh_bounded_rules() -> None:
    """Use edited local rules on rerun without touching D12 or user state."""
    global ConversationEngine, build_conversation_engine
    stamp = (ROOT / "ai" / "bounded_answers.py").stat().st_mtime_ns
    if getattr(conversation_module, "_bounded_rules_stamp", None) != stamp:
        conversation_module.interpret_bounded_answer = importlib.reload(bounded_rules).interpret_bounded_answer
        conversation_module._bounded_rules_stamp = stamp
    engine_stamp = (ROOT / "ai" / "conversation_engine.py").stat().st_mtime_ns
    if getattr(conversation_module, "_ENGINE_SOURCE_STAMP", None) != engine_stamp:
        ConversationEngine = importlib.reload(conversation_module).ConversationEngine
        build_conversation_engine = importlib.reload(factory_module).build_conversation_engine


def initialise() -> None:
    if "iw_engine" not in st.session_state:
        with st.spinner("Preparing your conversation…"):
            st.session_state.iw_engine = build_conversation_engine()
    elif type(st.session_state.iw_engine) is not ConversationEngine:
        # Streamlit reloads changed modules but keeps session objects alive.
        # Move the existing instance data into the current engine class so a
        # rule fix is usable without resetting answers or retraining D12.
        refreshed = object.__new__(ConversationEngine)
        refreshed.__dict__.update(vars(st.session_state.iw_engine))
        st.session_state.iw_engine = refreshed
    engine = st.session_state.iw_engine
    # Existing sessions keep their answers and pending confirmation. Attach
    # the all-question models only for subsequent replies; do not relabel past
    # user-confirmed answers or silently replay their conversation.
    if getattr(engine, "interpretation_mode", None) != "legacy" and getattr(engine, "question_resolver", None) is None:
        from ai.all_question_ml import build_all_question_ml_resolver
        engine.question_resolver = build_all_question_ml_resolver(engine.spec, ROOT / "data/nlp/all_questions")
        engine.interpretation_mode = "ml"
    if not hasattr(engine, "question_resolver"):
        engine.question_resolver = None
    defaults = {"iw_messages": [], "iw_last_response": None,
                "iw_finished_incomplete": False, "iw_save_result": None,
                "iw_unsaved": False, "iw_question_token": None,
                "iw_review_token": None, "iw_needs_correction": False,
                "iw_help_pending": False, "iw_material_limit_owner": None,
                "iw_large_text": False, "iw_latest_classification": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if not st.session_state.iw_messages:
        add_message("assistant", "Hi, I’m InvestWise. We’ll talk through what this money means to you, one question at a time. Answer in your own words; it’s okay to be unsure or ask for help. I’ll check my understanding with you before recording an answer.\n\nPlease use a fictional situation for this demo.")


def restart() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("iw_"):
            del st.session_state[key]
    initialise()
    st.rerun()


def ensure_question(engine: Any) -> None:
    q = engine.current_question
    if q and q.id == "D11" and "understanding_hold" in engine.state.active_holds:
        # The specific teach-back prompt already appears in the dialogue.
        return
    if q and (q.id, engine.state.profile_version) != st.session_state.iw_question_token:
        add_message("assistant", q.question, kind="question", qid=q.id,
                    position=engine.state.question_order.index(q.id) + 1)
        st.session_state.iw_question_token = (q.id, engine.state.profile_version)


def receive_result(result: dict[str, Any]) -> None:
    st.session_state.iw_last_response = result
    action = result.get("action")
    if action == "confirm":
        detail = context_text(result.get("context_details"))
        detail_line = f"\n\nI also understood: {detail}." if detail else ""
        add_message("assistant", f"Let me check: **{result['candidate_option_label']}**.{detail_line}\n\nHave I understood you correctly? You can say yes, correct it in your own words, or tell me you’re unsure.")
    elif action == "confirm_check":
        add_message("assistant", result["confirmation_text"] + "\n\nHave I understood this check correctly? Please confirm or correct it.")
    elif action == "material_check_resolved":
        st.session_state.iw_review_token = None
        add_message("assistant", "Thank you. I’ve confirmed that check. Please review the updated summary before deciding whether it can be saved.")
    elif action == "explain":
        st.session_state.iw_help_pending = True
        latest_user = next((message["text"] for message in reversed(st.session_state.iw_messages) if message["role"] == "user"), "")
        definition = term_explanation(result.get("question_id"), latest_user)
        if definition:
            st.session_state.iw_help_pending = False
            add_message("assistant", definition + "\n\n" + EXPLANATIONS[result["question_id"]])
        else:
            add_message("assistant", result.get("help_prompt") or "Would you like me to explain the question and choices, or do you understand them but cannot answer yet?")
    elif action in {"care_pause", "pace_pause"}:
        message = result.get("pacing_text") or "Thank you for telling me. We can take this at your pace. Would you like to continue, pause or stop?"
        if result.get("candidate_option_label"):
            message += "\n\nI have a possible answer in mind, but I will not record it unless you choose to continue and confirm it."
        add_message("assistant", message)
    elif action == "support_before_confirmation":
        message = result.get("help_prompt") or "I can explain this question before you decide whether my interpretation is right. Would you like an explanation, to continue, or to pause?"
        if st.session_state.iw_large_text:
            message = "I’ve increased the chat text size. " + message
        if result.get("candidate_option_label"):
            message += "\n\nI have a possible interpretation, but it has not been recorded."
        add_message("assistant", message)
    elif action == "understanding_hold":
        explanation = result.get("explanation") or "Investments can lose some or all of the original money permanently; recovery is not guaranteed."
        followup = result.get("followup") or "In your own words, what could happen to the original money?"
        add_message("assistant", f"{explanation}\n\n{followup}")
    elif action in {"clarify", "followup"}:
        if result.get("followup"):
            message = str(result["followup"])
        elif result.get("reason") == "explicit_durations_cross_d12_boundaries":
            message = "You mentioned different timeframes. What is the earliest point when you may realistically need any of this same money?"
        elif result.get("bucket") == "Undecided":
            message = "It’s okay to be unsure. I won’t choose for you. What detail are you still unsure about? You can also pause or leave this unresolved."
        else:
            message = "I haven’t understood that clearly enough to record an answer. Could you tell me a little more about what you mean, or which part of the question you’d like explained? You don’t need to use the wording of the choices."
        add_message("assistant", message)
    elif action == "finish_incomplete_or_pause":
        if st.session_state.iw_engine.current_question is None:
            st.session_state.iw_material_limit_owner = result.get("counter_owner") or result.get("question_id")
        add_message("assistant", "We don’t have to decide this now. I still haven’t understood this point clearly enough to confirm it, and I don’t want to guess. Would you like to pause or finish with this part incomplete?\n\nIn a live service, this could request human support. This demo simulates that step; no live agent has been contacted.")
    elif action == "safety_stop":
        add_message("assistant", "I’m sorry you’re going through this. I’ve stopped the financial questions. If you may be in immediate danger, contact your local emergency services now or someone you trust.\n\n**Demo: a human-support handover is simulated. No live agent has been contacted.**")
    elif action == "stop":
        add_message("assistant", "I’ve stopped this conversation. No further financial questions will be asked, and no completed profile has been saved. You can start a new fictional demo when you choose.")
    elif action in {"pause", "paused"}:
        add_message("assistant", "We can take a break. Your place in this conversation is kept in this session. Say ‘resume’ when you’re ready.")
    elif action == "resume":
        add_message("assistant", "Welcome back. We’ll carry on from where you paused.")
        if result.get("candidate_option_label"):
            detail = context_text(result.get("context_details"))
            detail_line = f"\n\nI also understood: {detail}." if detail else ""
            add_message("assistant", f"My proposed understanding is still **{result['candidate_option_label']}**.{detail_line}\n\nIs that right?")
    elif action == "request_correction_target":
        st.session_state.iw_needs_correction = True
        add_message("assistant", "Of course. Which earlier answer would you like to change? You can name its question number or use the answer list below.")
    elif action == "continue" and result.get("handled_as") == "declined":
        add_message("assistant", "That’s okay. I’ll show this answer as not provided. A missing answer needed for the financial profile will keep it incomplete.")


def submit(text: str, display_text: str | None = None, *, explicit_option_id: str | None = None) -> None:
    add_message("user", display_text or text)
    engine = st.session_state.iw_engine
    audit_start = len(engine.state.audit)
    result = engine.submit_answer(text, explicit_option_id=explicit_option_id)
    new_events = engine.state.audit[audit_start:]
    interpretation = next(
        (event for event in reversed(new_events)
         if event.event_type in {"ml_interpretation", "pre_model_control"}),
        None,
    )
    if interpretation is not None:
        st.session_state.iw_latest_classification = {
            "event_type": interpretation.event_type,
            "question_id": interpretation.question_id,
            "payload": dict(interpretation.payload),
            "result": dict(result),
        }
    elif result.get("bucket") or result.get("action") == "confirm":
        st.session_state.iw_latest_classification = {
            "event_type": "deterministic_interpretation",
            "question_id": result.get("question_id"),
            "payload": {"semantic_bucket": result.get("bucket") or "Clarity"},
            "result": dict(result),
        }
    receive_result(result)
    st.rerun()


def confirm_candidate(accepted: bool, user_text: str | None = None) -> None:
    add_message("user", user_text or ("Yes, that’s right." if accepted else "No, that’s not quite right."))
    result = st.session_state.iw_engine.confirm_current(accepted)
    st.session_state.iw_last_response = result
    if result.get("action") == "understanding_hold":
        receive_result(result)
    else:
        add_message("assistant", "Thank you. I’ve confirmed that answer." if accepted else "Thanks for correcting me. I haven’t recorded that interpretation. Tell me what you meant in your own words.")
    st.session_state.iw_help_pending = False
    st.rerun()


def correct(question_id: str) -> None:
    result = st.session_state.iw_engine.correct_answer(question_id)
    st.session_state.iw_last_response = result
    st.session_state.iw_finished_incomplete = False
    st.session_state.iw_save_result = None
    st.session_state.iw_unsaved = False
    st.session_state.iw_needs_correction = False
    st.session_state.iw_help_pending = False
    add_message("assistant", "Let’s update that answer. I’ll ask you to confirm the revised meaning, then show you the latest full summary again before any save.")
    st.rerun()


def explain_current() -> None:
    engine = st.session_state.iw_engine
    q = engine.current_question
    if q:
        add_message("assistant", EXPLANATIONS[q.id])
    elif engine.material_check_prompt():
        add_message("assistant", "I’m checking a missing financial fact before completing your profile. Your confirmed answers stay as given unless you correct them.\n\n" + engine.material_check_prompt())
    st.session_state.iw_help_pending = False
    st.rerun()


def accuracy(accepted: bool, text: str | None = None) -> None:
    add_message("user", text or ("Yes, the whole summary is accurate." if accepted else "It’s a little different."))
    result = st.session_state.iw_engine.confirm_final_accuracy(accepted)
    st.session_state.iw_last_response = result
    if result["accepted"]:
        add_message("assistant", "Thank you. May I save this exact confirmed version of your profile in this demo session? You can say yes, no, or that you’re unsure. Confirming accuracy alone has not saved it.")
    else:
        st.session_state.iw_needs_correction = True
        add_message("assistant", "Let’s get it right. Which answer is different? Use its question number or the answer list below. Nothing will be saved from this unconfirmed summary.")
    st.rerun()


def save_consent(accepted: bool, text: str | None = None) -> None:
    engine = st.session_state.iw_engine
    add_message("user", text or ("Yes, save this version." if accepted else "No, leave it unsaved."))
    result = engine.give_save_consent(accepted)
    st.session_state.iw_last_response = result
    if accepted and result["accepted"]:
        saved = engine.save()
        st.session_state.iw_save_result = saved
        add_message("assistant", "Your confirmed profile has been saved in this demo session. Thank you for reviewing it with me." if saved.get("saved") else "The profile could not be saved. Your answers remain here for review.")
    else:
        st.session_state.iw_unsaved = True
        add_message("assistant", "Understood. Your profile has not been saved. We can finish here.")
    st.rerun()


def handle_message(text: str) -> None:
    engine = st.session_state.iw_engine
    text = text.strip().replace("’", "'")
    if re.search(r"\b(?:larger text|make the text (?:bigger|larger)|enlarge the text)\b", text, re.I):
        st.session_state.iw_large_text = True
    control = check_pre_model_controls(text)
    named_change = re.fullmatch(r"(?:please\s+)?(?:(?:i(?:'d| would) like to|i want to)\s+)?(?:change|correct|update)\s+(?:(?:my|the)\s+answer\s+(?:to|for)\s+)?(?:d|q|question\s*)(1[0-5]|[1-9])[.!]*", text, re.I)
    if engine.current_question is None and named_change:
        add_message("user", text)
        correct(f"D{named_change.group(1)}")
    if control.action == "correction_request" and engine.current_question is None:
        engine.confirm_final_accuracy(False)
        st.session_state.iw_needs_correction = True
        add_message("user", text)
        add_message("assistant", "Which earlier answer should we change? You can name the question number or use the answer list below.")
        st.rerun()
    if control.matched and control.action in {"safety_stop", "stop", "pause", "pace_answer", "care_pause", "correction_request", "decline", "support_control", "support_answer"}:
        submit(text)
    if (engine.state.paused
            and {"care_pacing", "support_request"}.intersection(engine.state.active_holds)
            and control.action == "explain"):
        add_message("user", text)
        explain_current()
    if engine.state.paused:
        submit(text)
    if st.session_state.iw_needs_correction:
        add_message("user", text)
        match = re.search(r"\b(?:d|q|question\s*)(1[0-5]|[1-9])\b", text, re.I)
        if match:
            correct(f"D{match.group(1)}")
        add_message("assistant", "Which answer should I change? You can say ‘question 12’, for example, or use the list below.")
        st.rerun()
    if control.action == "explain" and engine.current_question is None:
        submit(text)
    if engine.current_question is None:
        previous_version = engine.state.profile_version
        review_result = engine.submit_answer(text)
        if review_result.get("action") != "final_review":
            add_message("user", text)
            receive_result(review_result)
            st.rerun()
        if engine.state.profile_version != previous_version:
            add_message("user", text)
            add_message("assistant", "You’ve added a new detail about needing this same money. I’ll keep your earlier answers visible, but we need to check this before confirming or saving the profile.")
            st.session_state.iw_review_token = None
            st.rerun()
    if engine.current_question:
        record = engine.state.answers[engine.current_question.id]
        if record.status == AnswerStatus.CANDIDATE:
            if YES.fullmatch(text):
                confirm_candidate(True, text)
            if NO.fullmatch(text):
                confirm_candidate(False, text)
            if control.action in {"explain", "support_control", "support_answer"}:
                submit(text)
            engine.confirm_current(False)
        if st.session_state.iw_help_pending and re.fullmatch(r"(?:please )?(?:explain(?: the question(?: and choices)?)?|i need an explanation|what does (?:it|this) mean)[.!?]*", text, re.I):
            add_message("user", text)
            explain_current()
        submit(text)
    elif engine.state.active_holds:
        add_message("user", text)
        if "d14_d15_tension_unacknowledged" in engine.state.active_holds and re.fullmatch(r"(?:i understand[;,]? )?(?:keep both(?: answers)?|retain both(?: answers)?)[.!]*",text,re.I):
            engine.acknowledge_preference_comfort_tension()
            st.session_state.iw_review_token = None
            add_message("assistant", "I’ll retain both answers as given. Please review the current summary.")
        elif getattr(engine,"pending_material_check",None) and (YES.fullmatch(text) or NO.fullmatch(text)):
            receive_result(engine.confirm_material_check(bool(YES.fullmatch(text))))
        elif hasattr(engine,"resolve_material_check"):
            receive_result(engine.resolve_material_check(text))
        else:
            add_message("assistant", "There is still an unresolved check. You can correct an answer using the list, or leave the profile incomplete. I won’t assume how to resolve it.")
        st.rerun()
    elif engine.state.final_accuracy_version != engine.state.profile_version:
        if YES.fullmatch(text) or FINAL_ACCURACY_YES.fullmatch(text):
            accuracy(True, text)
        if NO.fullmatch(text):
            accuracy(False, text)
    else:
        if SAVE_YES.fullmatch(text):
            save_consent(True, text)
        if SAVE_NO.fullmatch(text):
            save_consent(False, text)
        if NO.fullmatch(text) or text.lower().startswith("no,"):
            accuracy(False, text)
    add_message("user", text)
    add_message("assistant", "I won’t assume that means yes. You can confirm, tell me what needs changing, or leave it unconfirmed.")
    st.rerun()


def render_header(engine: Any) -> None:
    st.markdown("""<style>
    .stMainBlockContainer {max-width: 820px; padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stChatMessage"] {border-radius: 18px; padding: 1rem 1.1rem; margin-bottom: .75rem; border: 1px solid rgba(135,145,165,.16); background: rgba(135,145,165,.08);}
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {background: rgba(56,156,111,.16); border-color: rgba(56,156,111,.3); width: 90%; margin-left: auto;}
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {width: 95%; margin-right: auto;}
    [data-testid="stChatInput"] {border-radius: 20px;}
    .stButton button {border-radius: 18px;}
    h1 {font-size: 1.85rem !important; letter-spacing: -.04em;}
    @media (max-width: 640px) {
      .st-key-iw_thread {height: 330px !important;}
      .stMainBlockContainer {padding-top: 1rem;}
    }
    </style>""", unsafe_allow_html=True)
    if st.session_state.iw_large_text:
        st.markdown('<style>[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {font-size: 1.25rem !important; line-height: 1.65;}</style>', unsafe_allow_html=True)
    st.title("🌿 InvestWise")
    st.caption("A careful conversation · fictional situations only")
    handled = sum(r.status in {AnswerStatus.CONFIRMED, AnswerStatus.DECLINED} for r in engine.state.answers.values())
    status = "Paused" if engine.state.paused else "Conversation stopped" if engine.state.stopped_for_safety else f"{handled} of 15 answers reviewed · take your time"
    st.caption(status)


def render_demo_details(engine: Any) -> None:
    with st.expander("Conversation menu and demo details"):
        st.button("Start a new conversation", on_click=restart)
        with st.expander("About this prototype"):
            st.write("All 15 questions interpret free text using TF-IDF and Logistic Regression, with conformal prediction for both the three dialogue buckets and the question’s fixed options. A single supported interpretation is proposed for your confirmation; uncertain predictions stay unresolved. Safety, business consistency and consent checks remain explicit safeguards. The models use labelled fictional examples and may still make errors; this prototype has not established real-world accuracy.")
            st.write("Answers and the dialogue remain in this local demo session. Save consent is separate from confirming accuracy. Human handovers are simulated. No investment advice is given.")
        with st.expander("Demo diagnostics"):
            current_id = engine.state.current_question_id or "D12"
            record = engine.state.answers[current_id]
            st.json({"profile_version": engine.state.profile_version,
                     "current_question": engine.state.current_question_id,
                     "answer": asdict(record), "interpretation_mode": getattr(engine, "interpretation_mode", "legacy"),
                     "model_source_hash": getattr(getattr(engine, "question_resolver", None), "source_hash", None),
                     "active_holds": sorted(engine.state.active_holds),
                     "accuracy_version": engine.state.final_accuracy_version,
                     "save_consent_version": engine.state.save_consent_version})
            st.write("Session audit")
            st.json([asdict(event) for event in engine.state.audit])


def _display_label(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).replace("_", " ").strip().title()


def render_latest_classification() -> None:
    """Show a simple owner result and a separate technical trace."""
    trace = st.session_state.iw_latest_classification
    if not trace:
        return

    payload = trace["payload"]
    result = trace["result"]
    is_control = trace["event_type"] == "pre_model_control"
    is_model = trace["event_type"] == "ml_interpretation"
    bucket = result.get("bucket") if is_control else payload.get("semantic_bucket")
    owner_label = {
        "Clarity": "Clarity",
        "Confusion": "Confused",
        "Undecided": "Undecided",
    }.get(bucket)

    # Never relabel model uncertainty as a user meaning. The owner card is
    # shown only when one of the three dialogue meanings is established.
    if owner_label:
        with st.container(border=True):
            st.caption("Conversation classification")
            st.subheader(owner_label)

    action = result.get("action")
    candidate = result.get("candidate_option_label") or result.get("candidate_option_id")
    if is_control:
        developer_trace = {
            "Control detected": _display_label(payload.get("reason") or payload.get("flag") or payload.get("action")),
            "Model called": "No",
            "Financial candidate": None,
            "Action": _display_label(action),
        }
    elif is_model:
        developer_trace = {
            "Control detected": None,
            "Model called": "Yes",
            "Dialogue bucket": bucket,
            "Intent set": payload.get("intent_set"),
            "Option set": payload.get("option_set"),
            "Financial candidate": candidate,
            "Action": _display_label(action),
            "Model uncertain": payload.get("model_uncertain", False),
            "Reason": payload.get("reason"),
        }
    else:
        developer_trace = {
            "Control detected": None,
            "Model called": "No",
            "Financial candidate": candidate,
            "Action": _display_label(action),
            "Reason": "Deterministic completion of supplied facts",
        }
    with st.expander("Developer classification trace"):
        st.json(developer_trace)


def render_final_profile_status(engine: Any) -> None:
    if st.session_state.iw_finished_incomplete:
        with st.container(border=True):
            st.caption("Final profile status")
            st.subheader("Incomplete — no final classification produced")
            st.write("At least one answer remained unresolved, so the safety gate did not create or save a financial profile.")
        return
    if engine.current_question is not None:
        return

    report = engine.completion_report()
    if not report["eligible_for_final_accuracy_confirmation"]:
        return
    answers = engine.state.answers
    with st.container(border=True):
        st.caption("Final profile classifications")
        st.write({
            "Time horizon": answers["D12"].selected_option_label,
            "Capacity for loss": answers["D13"].selected_option_label,
            "Attitude to risk": answers["D14"].selected_option_label,
        })
        st.caption("Awaiting the user's confirmation of the complete summary." if engine.state.final_accuracy_version != engine.state.profile_version else "Confirmed by the user.")


def render_transcript(engine: Any) -> None:
    with st.container(height=330, border=False, autoscroll=True, key="iw_thread"):
        for item in st.session_state.iw_messages:
            with st.chat_message(item["role"], avatar="🌿" if item["role"] == "assistant" else "🙂"):
                if item.get("kind") == "question":
                    st.caption(f"Question {item['position']} of 15")
                st.markdown(item["text"])
                if item.get("kind") == "question":
                    q = engine.spec.question(item["qid"])
                    with st.expander("Possible meanings · answer in your own words"):
                        for option in q.options:
                            st.markdown(f"- {option.label}")
                        st.caption("You can also say you’re unsure, ask for help or leave this unanswered.")


def render_correction_list(engine: Any) -> None:
    with st.expander("Review or change an earlier answer", expanded=st.session_state.iw_needs_correction):
        choices = [qid for qid in engine.state.question_order if engine.state.answers[qid].status in {AnswerStatus.CONFIRMED, AnswerStatus.DECLINED}]
        if not choices:
            st.caption("No earlier answer has been confirmed yet.")
            return
        selected = st.selectbox("Which answer?", choices, format_func=lambda qid: f"{qid} · {engine.spec.question(qid).topic}: {engine.state.answers[qid].selected_option_label or 'Not provided'}")
        if st.button("Change this answer"):
            add_message("user", f"I’d like to change {selected}.")
            correct(selected)


def render_current_actions(engine: Any) -> None:
    q = engine.current_question
    record = engine.state.answers[q.id]
    if record.status == AnswerStatus.CANDIDATE:
        with st.container(horizontal=True, wrap=True):
            if st.button("Yes, that’s right", type="primary"):
                confirm_candidate(True)
            if st.button("Not quite"):
                confirm_candidate(False)
        st.caption("Or reply below in your own words. This interpretation is awaiting confirmation.")
    elif q.id == "D11" and "understanding_hold" in engine.state.active_holds:
        st.caption("Please explain the possibility of permanent loss in your own words before we continue. An option shortcut cannot complete this check.")
    elif st.session_state.iw_help_pending:
        with st.container(horizontal=True, wrap=True):
            if st.button("Explain the question"):
                add_message("user", "Please explain the question.")
                explain_current()
            if st.button("I understand, but I’m unsure"):
                st.session_state.iw_help_pending = False
                submit("not sure", "I understand, but I’m unsure.")
    else:
        with st.expander("Answer shortcuts · optional"):
            st.caption("You can always type instead. No option is preselected.")
            cols = st.columns(2)
            for index, option in enumerate(q.options):
                if cols[index % 2].button(option.label, key=f"iw_option_{q.id}_{option.id}", use_container_width=True):
                    submit(option.id, option.label, explicit_option_id=option.id)
    with st.container(horizontal=True, wrap=True):
        if st.button("Help"):
            submit("I need help")
        if st.button("I’m unsure"):
            submit("not sure")
        if st.button("Pause"):
            submit("pause")
        if st.button("Skip"):
            submit("prefer not to answer")


def summary_text(engine: Any) -> str:
    playback = engine.final_playback()
    rows = {row["question_id"]: row for row in playback["answers"]}

    def label(qid: str) -> str:
        row = rows[qid]
        base = row["selected_option_label"] or ("Not provided" if row["status"] == AnswerStatus.DECLINED.value else "Unresolved")
        detail = context_text(row.get("context_details")) if row["status"] == AnswerStatus.CONFIRMED.value else ""
        return f"{base} ({detail})" if detail else base
    conclusions = "\n\n".join([
        f"**Time horizon — earliest realistic need:** {label('D12')}",
        f"**Capacity for loss — practical effect in the 20% example:** {label('D13')}",
        f"**Stated attitude to risk:** {label('D14')}",
        f"**Separate comfort qualifier:** {label('D15')}"])
    answers = "\n".join(
        f"- **{row['question_id']} · {engine.spec.question(row['question_id']).topic} "
        f"({row['status']}):** {label(row['question_id'])}"
        for row in playback["answers"]
    )
    notes = []
    if engine.state.active_holds:
        notes.append("Some checks remain unresolved, so this profile is incomplete.")
    if (engine.state.answers["D14"].selected_option_id == "D14_GREATER"
            and engine.state.answers["D15"].selected_option_id == "D15_UNACCEPTABLE"
            and any(event.event_type == "tension_acknowledged" for event in engine.state.audit)):
        notes.append("Your greater-growth preference and discomfort with the example fall are both retained as stated.")
    note_line = "\n\n**Context and checks:** " + " ".join(notes) if notes else ""
    return ("Here is my understanding of the money you described. Please review all the answers.\n\n"
            f"**Money and purpose:** {label('D1')}\n\n"
            f"**Existing amount and named currency, if provided:** {label('D3')}\n\n"
            f"{conclusions}\n\nThe 20% example describes practical impact; it does not establish the maximum loss you could afford."
            f"{note_line}\n\n**Your answers**\n\n{answers}")


def ensure_review(engine: Any) -> None:
    if st.session_state.iw_review_token != engine.state.profile_version:
        add_message("assistant", summary_text(engine))
        report = engine.completion_report()
        if report["eligible_for_final_accuracy_confirmation"]:
            add_message("assistant", "Is this exact summary accurate? You can confirm it, tell me what’s different, or leave it unconfirmed.")
        else:
            add_message("assistant", "There are still missing answers or checks to resolve, so this profile is incomplete. I won’t save it as a completed profile.")
            if hasattr(engine,"material_check_prompt"):
                prompt = engine.material_check_prompt()
                if prompt:
                    add_message("assistant", prompt)
        st.session_state.iw_review_token = engine.state.profile_version


def render_review_actions(engine: Any) -> None:
    report = engine.completion_report()
    holds = set(report["active_holds"])
    if "d14_d15_tension_unacknowledged" in holds:
        st.write("Your stated preference and comfort with the example fall differ. Both answers can be retained without combining them.")
        if st.button("I understand; keep both answers"):
            add_message("user", "I understand the difference; keep both answers.")
            engine.acknowledge_preference_comfort_tension()
            add_message("assistant", "I’ll retain both as given. Please review the summary as a whole.")
            st.rerun()
        return
    if getattr(engine,"pending_material_check",None):
        a,b=st.columns(2)
        if a.button("Yes, that check is right",type="primary"):
            add_message("user","Yes, that check is right.")
            receive_result(engine.confirm_material_check(True))
            st.rerun()
        if b.button("No, correct that check"):
            add_message("user","No, correct that check.")
            receive_result(engine.confirm_material_check(False))
            st.rerun()
    if holds.intersection({"same_money_dependency", "emergency_long_horizon_dependency", "living_costs_horizon_dependency", "same_pot_backup_conflict", "usable_backup_check", "need_before_access", "essential_funding_timing"}):
        st.caption("Answer the current check in your own words. If the same money is needed sooner, correct that answer; if you’re unsure, leave the check unresolved.")
    if not report["eligible_for_final_accuracy_confirmation"]:
        if st.button("Finish incomplete"):
            st.session_state.iw_finished_incomplete = True
            add_message("assistant", "We’ll leave this profile incomplete and unsaved. You don’t have to decide now.")
            st.rerun()
        return
    if engine.state.saved_version == engine.state.profile_version or st.session_state.iw_unsaved:
        return
    a, b = st.columns(2)
    if engine.state.final_accuracy_version != engine.state.profile_version:
        if a.button("Yes, the summary is accurate", type="primary", use_container_width=True):
            accuracy(True)
        if b.button("It’s a little different", use_container_width=True):
            accuracy(False)
    else:
        if a.button("Yes, save this version", type="primary", use_container_width=True):
            save_consent(True)
        if b.button("No, leave it unsaved", use_container_width=True):
            save_consent(False)


def main() -> None:
    refresh_bounded_rules()
    initialise()
    engine = st.session_state.iw_engine
    render_header(engine)
    if (not engine.state.stopped_for_safety and not st.session_state.iw_finished_incomplete
            and "user_stop" not in engine.state.active_holds):
        if engine.current_question:
            ensure_question(engine)
        else:
            ensure_review(engine)
    render_transcript(engine)
    render_latest_classification()
    render_final_profile_status(engine)
    if engine.state.stopped_for_safety:
        st.caption("Financial questions and saving have been stopped.")
        render_demo_details(engine)
        return
    if "user_stop" in engine.state.active_holds:
        st.caption("This conversation has been stopped. No completed profile has been saved.")
        render_demo_details(engine)
        return
    if st.session_state.iw_finished_incomplete:
        render_demo_details(engine)
        return
    if engine.state.paused:
        if {"care_pacing", "support_request"}.intersection(engine.state.active_holds):
            with st.container(horizontal=True, wrap=True):
                if st.button("Continue when ready", type="primary"):
                    add_message("user", "I’m ready to continue.")
                    receive_result(engine.resume())
                    st.rerun()
                if st.button("Keep paused"):
                    submit("pause")
                if st.button("Stop this conversation"):
                    submit("stop")
                if st.button("Explain the current question"):
                    add_message("user", "Please explain the current question.")
                    explain_current()
        elif st.button("Resume conversation", type="primary"):
            add_message("user", "Resume.")
            receive_result(engine.resume())
            st.rerun()
    elif engine.current_question and engine.state.answers[engine.current_question.id].clarification_turns >= 3 and engine.state.answers[engine.current_question.id].status in {AnswerStatus.UNDECIDED, AnswerStatus.CONFUSION, AnswerStatus.PENDING}:
        a, b = st.columns(2)
        if a.button("Pause conversation", use_container_width=True):
            submit("pause")
        if b.button("Finish incomplete", use_container_width=True):
            st.session_state.iw_finished_incomplete = True
            add_message("user", "Finish incomplete.")
            add_message("assistant", "We’ll leave this profile incomplete and unsaved. You don’t have to decide now.")
            st.rerun()
        if engine.current_question.id in OPTIONAL_CONTEXT and st.button("Leave this contextual answer unanswered"):
            submit("prefer not to answer")
        render_demo_details(engine)
        return
    elif (engine.current_question is None and engine.state.active_holds
          and st.session_state.iw_material_limit_owner is not None):
        if st.button("Pause conversation"):
            submit("pause")
        if st.button("Finish incomplete"):
            st.session_state.iw_finished_incomplete = True
            add_message("user", "Finish incomplete.")
            add_message("assistant", "We’ll leave this profile incomplete and unsaved. You don’t have to decide now.")
            st.rerun()
        render_correction_list(engine)
        render_demo_details(engine)
        return
    elif not st.session_state.iw_needs_correction:
        if engine.current_question:
            render_current_actions(engine)
        else:
            render_review_actions(engine)
    render_correction_list(engine)
    render_demo_details(engine)
    if engine.state.saved_version == engine.state.profile_version or st.session_state.iw_unsaved:
        return
    text = st.chat_input("Reply in your own words…", key="iw_chat_input")
    if text:
        handle_message(text)


if __name__ == "__main__":
    main()
