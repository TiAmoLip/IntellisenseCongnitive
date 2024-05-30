import sys
from loguru import logger
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

def get_logger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


def ptb_tokenize(key_to_captions):
    captions_for_image = {}
    for key, caps in key_to_captions.items():
        captions_for_image[key] = []
        for idx, cap in enumerate(caps):
            captions_for_image[key].append({
                # "image_id": key
                # "id": idx,
                "caption": cap
            })
    tokenizer = PTBTokenizer()
    key_to_captions = tokenizer.tokenize(captions_for_image)
    return key_to_captions




def bleu_score_fn(method_no: int = 4, ref_type='corpus'):
    """
    :param method_no:
    :param ref_type: 'corpus' or 'sentence'
    :return: bleu score
    """
    smoothing_method = getattr(SmoothingFunction(), f'method{method_no}')

    def bleu_score_corpus(reference_corpus: list, candidate_corpus: list, n: int = 4):
        """
        :param reference_corpus: [b, 5, var_len]
        :param candidate_corpus: [b, var_len]
        :param n: size of n-gram
        """
        weights = [1 / n] * n
        return corpus_bleu(reference_corpus, candidate_corpus,
                           smoothing_function=smoothing_method, weights=weights)

    def bleu_score_sentence(reference_sentences: list, candidate_sentence: list, n: int = 4):
        """
        :param reference_sentences: [5, var_len]
        :param candidate_sentence: [var_len]
        :param n: size of n-gram
        """
        weights = [1 / n] * n
        return sentence_bleu(reference_sentences, candidate_sentence,
                             smoothing_function=smoothing_method, weights=weights)

    if ref_type == 'corpus':
        return bleu_score_corpus
    elif ref_type == 'sentence':
        return bleu_score_sentence
