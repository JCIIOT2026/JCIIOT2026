import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

_tokenizer = None
_lang_emb_model = None
_tz = None

LANG_EMB_OBS_KEY = "lang_emb"


def _lazy_init():
    global _tokenizer, _lang_emb_model, _tz
    if _tz is not None:
        return
    from transformers import CLIPTextModelWithProjection, AutoTokenizer

    _tokenizer = "openai/clip-vit-large-patch14"
    _lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
        _tokenizer,
        cache_dir=os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))
    ).eval()
    _tz = AutoTokenizer.from_pretrained(_tokenizer, TOKENIZERS_PARALLELISM=True)


def get_lang_emb(lang):
    if lang is None:
        return None

    _lazy_init()

    tokens = _tz(
        text=lang,
        add_special_tokens=True,
        max_length=25,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    lang_emb = _lang_emb_model(**tokens)['text_embeds'].detach()[0]
    return lang_emb

def get_lang_emb_shape():
    return list(get_lang_emb('dummy').shape)