"""Bridge upstream infer(custom_streamer=...) and stock DeepSeek-OCR infer().

Caller must serialize access to the model while generate is temporarily wrapped.
No model source files or downloaded checkpoints are modified.
"""
import inspect


def infer_with_streamer(model, tokenizer, streamer=None, stop_flag=None, **kwargs):
    if 'custom_streamer' in inspect.signature(model.infer).parameters:
        return model.infer(tokenizer, custom_streamer=streamer, **kwargs)
    if streamer is None:
        return model.infer(tokenizer, **kwargs)
    generate = model.generate
    had_override = 'generate' in model.__dict__
    previous = model.__dict__.get('generate')

    def streamed_generate(*args, **generation_kwargs):
        generation_kwargs['streamer'] = streamer
        if stop_flag is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList
            class StopRequested(StoppingCriteria):
                def __call__(self, input_ids, scores, **unused):
                    return bool(stop_flag.get('stop'))
            criteria = list(generation_kwargs.get('stopping_criteria') or [])
            generation_kwargs['stopping_criteria'] = StoppingCriteriaList(criteria + [StopRequested()])
        return generate(*args, **generation_kwargs)

    model.generate = streamed_generate
    try:
        return model.infer(tokenizer, **kwargs)
    finally:
        if had_override:
            model.generate = previous
        else:
            del model.generate
