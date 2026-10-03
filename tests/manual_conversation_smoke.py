from ai.conversation_factory import build_conversation_engine


engine = build_conversation_engine()


for question_id in engine.state.question_order:
    assert engine.state.current_question_id == question_id

    question = engine.spec.question(question_id)

    if question_id == "D12":
        answer = "I may realistically need this money in 7 years"
    else:
        # Use the first valid fixed option defined by v7.
        answer = question.options[0].id

    print()
    print("=" * 70)
    print("QUESTION", question_id)
    print(question.question)
    print("ANSWER", answer)

    response = engine.submit_answer(answer)

    print("RESPONSE")
    print(response)

    if response["action"] != "confirm":
        print()
        print("UNEXPECTED ACTION")
        print(response)
        break

    confirmation = engine.confirm_current(True)

    print("CONFIRMATION")
    print(confirmation)


print()
print("=" * 70)
print("FINAL PLAYBACK")
playback = engine.final_playback()
print(playback)


print()
print("=" * 70)
print("COMPLETION")
print(playback["completion"])


print()
print("=" * 70)
print("ACCURACY CONFIRMATION")
accuracy = engine.confirm_final_accuracy(True)
print(accuracy)


print()
print("=" * 70)
print("SAVE CONSENT")
consent = engine.give_save_consent(True)
print(consent)


print()
print("=" * 70)
print("SAVE")
saved = engine.save()
print(saved)