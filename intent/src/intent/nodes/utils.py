# author: Steeve LAQUITAINE


# either ? or ! or .
SENT_TYPE_PATTN = re.compile(r"[\?\!\.]")


def classify_sentence_type(sentences):
    """
    Classify sentence type
    """
    sent_type = []
    for sent in sentences:
        out = SENT_TYPE_PATTN.findall(sent)
        sent_type.append(
            [
                "ask" if ix == "?" else "wish-or-excl" if ix == "!" else "state"
                for ix in out
            ]
        )
    return sent_type


def detect_sentence_type(df, sent_type: str):
    """
    Detect sentence types

    parameters
    ----------
    sent_type: str
        'state', 'ask', 'wish-excl' 
    """
    return sent_type in df
