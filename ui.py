"""Chainlit UI for nano-00-agent.

Run: chainlit run ui.py -w --port 8002
"""
import json
import uuid

import chainlit as cl
import httpx

API_BASE = "http://localhost:8001"


@cl.on_chat_start
async def on_start():
    """Initialize conversation."""
    conv_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conv_id)
    cl.user_session.set("turn_count", 0)

    await cl.Message(
        content="**nano-00-agent** 에 오신 걸 환영합니다!\n\n"
        "💬 자유롭게 질문하세요\n"
        "📄 파일을 첨부하면 지식으로 등록됩니다 (PDF, CSV, XLSX)\n"
        "🔍 등록된 지식을 기반으로 검색·분석·요약이 가능합니다\n\n"
        f"*Conversation ID: `{conv_id[:8]}...`*",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message — route through pipeline API."""
    conv_id = cl.user_session.get("conversation_id")
    turn = cl.user_session.get("turn_count", 0) + 1
    cl.user_session.set("turn_count", turn)

    # Handle file uploads
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                await _upload_file(element)

    # Send to pipeline API
    thinking_msg = cl.Message(content="")
    await thinking_msg.send()

    # Step 1: Goal generation indicator
    async with cl.Step(name="Goal 생성", type="llm") as step:
        step.input = message.content

    async with cl.Step(name="Intent 분류", type="llm") as step:
        step.input = f"Turn {turn}"

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{API_BASE}/chat",
                json={
                    "conversation_id": conv_id,
                    "message": message.content,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        await cl.Message(content=f"❌ API 에러: {e.response.status_code}\n```\n{e.response.text}\n```").send()
        return
    except Exception as e:
        await cl.Message(content=f"❌ 연결 실패: {e}\n\nAPI 서버가 `{API_BASE}`에서 실행 중인지 확인하세요.").send()
        return

    # Extract data
    response_text = data.get("response", "응답 없음")
    goal = data.get("goal", {})
    token_count = data.get("token_count", {})
    state = data.get("conversation_state", {})
    pending_hitl = data.get("pending_hitl")
    task_progress = data.get("task_progress")

    # Show Goal
    if goal and goal.get("final_objective"):
        async with cl.Step(name="Goal", type="tool") as step:
            step.output = (
                f"**목표**: {goal['final_objective']}\n"
                f"**진행**: {goal.get('progress_pct', 0)}%\n"
                f"**기준**: {', '.join(goal.get('success_criteria', []))}"
            )

    # Show tool calls / task progress
    if task_progress:
        async with cl.Step(name="태스크 진행", type="tool") as step:
            step.output = (
                f"상태: {task_progress.get('status')}\n"
                f"완료: {task_progress.get('completed_steps')}/{task_progress.get('total_steps')} 스텝"
            )

    # Show pointer resolution if any
    accumulated = state.get("accumulated_data", {})
    if accumulated:
        async with cl.Step(name=f"포인터 {len(accumulated)}개", type="retrieval") as step:
            lines = []
            for key, val in accumulated.items():
                if isinstance(val, dict) and "ptr" in val:
                    lines.append(f"- `{val['ptr'][:30]}...` → {val.get('desc', key)}")
            step.output = "\n".join(lines) if lines else "없음"

    # Show HITL if pending
    if pending_hitl:
        res = await cl.AskActionMessage(
            content=f"⚠️ **확인 필요**: {pending_hitl.get('description', '')}",
            actions=[
                cl.Action(name="confirm", payload={"value": "yes"}, label="✅ 진행"),
                cl.Action(name="cancel", payload={"value": "no"}, label="❌ 취소"),
            ],
        ).send()
        if res and res.get("value") == "yes":
            # Re-send with HITL confirmation
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp2 = await client.post(
                    f"{API_BASE}/chat",
                    json={
                        "conversation_id": conv_id,
                        "message": message.content,
                        "hitl_confirmation": {
                            "confirmed": True,
                            "action": pending_hitl.get("action"),
                        },
                    },
                )
                if resp2.status_code == 200:
                    data2 = resp2.json()
                    response_text = data2.get("response", response_text)

    # Build footer
    intent_chain = state.get("intent_chain", [])
    token_info = token_count.get("total_this_turn", 0)
    compress_flag = "⚠️ 압축 필요" if token_count.get("should_compress") else ""

    footer = f"\n\n---\n*Turn {turn} | 토큰: {token_info} {compress_flag} | 맥락: {len(intent_chain)}턴*"

    # Update thinking message with response
    thinking_msg.content = response_text + footer
    await thinking_msg.update()
