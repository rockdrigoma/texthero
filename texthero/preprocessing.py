"""
The texthero.preprocess module allow for efficient pre-processing of text-based Pandas Series and DataFrame.
"""

from gensim.sklearn_api.phrases import PhrasesTransformer
import re
import string
from typing import Optional, Set
import unicodedata

import numpy as np
import pandas as pd
import unidecode

from texthero import stopwords as _stopwords
from texthero._types import TokenSeries, TextSeries, InputSeries

from typing import List, Callable, Union

import emoji

import nltk
from nltk import word_tokenize 
from nltk.util import ngrams
nltk.download("punkt")

import pkg_resources
from symspellpy import SymSpell, Verbosity

import json

from difflib import SequenceMatcher
from itertools import zip_longest
import re

import random

# Ignore gensim annoying warnings
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")

with open("tropical_dic.json", "r") as file:
    tropical_dic = json.load(file)

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_es_82_765.txt"
bigram_path = "frequency_bigramdictionary_es_1Mnplus.txt"

# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

FIRST_INT = 1111111111
LAST_INT = 9999999999
PLACEHOLDERS_DICT = {}

@InputSeries(TextSeries)
def fillna(s: TextSeries) -> TextSeries:
    """
    Replaces not assigned values with empty string.


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["I'm", np.NaN, pd.NA, "You're"])
    >>> hero.fillna(s)
    0       I'm
    1          
    2          
    3    You're
    dtype: object
    """
    return s.fillna("").astype("str")


@InputSeries(TextSeries)
def lowercase(s: TextSeries) -> TextSeries:
    """
    Lowercase all texts in a series.

    
    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("This is NeW YoRk wIth upPer letters")
    >>> hero.lowercase(s)
    0    this is new york with upper letters
    dtype: object
    """
    return s.str.lower()


@InputSeries(TextSeries)
def replace_digits(s: TextSeries, symbols: str = " ", only_blocks=True) -> TextSeries:
    """
    Replace all digits with symbols.

    By default, only replaces "blocks" of digits, i.e tokens composed of only
    numbers.

    When `only_blocks` is set to ´False´, replaces all digits.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbols : str, optional, default=" "
        Symbols to replace

    only_blocks : bool, optional, default=True
        When set to False, replace all digits.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("1234 falcon9")
    >>> hero.preprocessing.replace_digits(s, "X")
    0    X falcon9
    dtype: object
    >>> hero.preprocessing.replace_digits(s, "X", only_blocks=False)
    0    X falconX
    dtype: object
    """

    if only_blocks:
        pattern = r"\b\d+\b"
        return s.str.replace(pattern, symbols)
    else:
        return s.str.replace(r"\d+", symbols)


@InputSeries(TextSeries)
def remove_digits(s: TextSeries, only_blocks=True) -> TextSeries:
    """
    Remove all digits and replaces them with a single space.

    By default, only remove "blocks" of digits. For instance, `1234 falcon9`
    becomes ` falcon9`.

    When the arguments `only_blocks` is set to ´False´, remove any digits.

    See also :meth:`replace_digits` to replace digits with another string.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    only_blocks : bool, optional, default=True
        Remove only blocks of digits.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("7ex7hero is fun 1111")
    >>> hero.preprocessing.remove_digits(s)
    0    7ex7hero is fun  
    dtype: object
    >>> hero.preprocessing.remove_digits(s, only_blocks=False)
    0     ex hero is fun  
    dtype: object
    """

    return replace_digits(s, " ", only_blocks)


@InputSeries(TextSeries)
def replace_punctuation(s: TextSeries, symbol: str = " ") -> TextSeries:
    """
    Replace all punctuation with a given symbol.

    Replace all punctuation from the given
    Pandas Series with a custom symbol. 
    It considers as punctuation characters all :data:`string.punctuation` 
    symbols `!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).`


    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbol : str, optional, default=" "
        Symbol to use as replacement for all string punctuation. 

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Finnaly.")
    >>> hero.replace_punctuation(s, " <PUNCT> ")
    0    Finnaly <PUNCT> 
    dtype: object
    """

    return s.str.replace(rf"([{string.punctuation}])+", symbol)


@InputSeries(TextSeries)
def remove_punctuation(s: TextSeries) -> TextSeries:
    """
    Replace all punctuation with a single space (" ").

    Remove all punctuation from the given Pandas Series and replace it
    with a single space. It considers as punctuation characters all
    :data:`string.punctuation` symbols `!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).`

    See also :meth:`replace_punctuation` to replace punctuation with a custom
    symbol.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Finnaly.")
    >>> hero.remove_punctuation(s)
    0    Finnaly 
    dtype: object
    """
    return replace_punctuation(s, " ")


def _remove_diacritics(text: str) -> str:
    """
    Remove diacritics and accents from one string.

    Examples
    --------
    >>> from texthero.preprocessing import _remove_diacritics
    >>> import pandas as pd
    >>> text = "Montréal, über, 12.89, Mère, Françoise, noël, 889, اِس, اُس"
    >>> _remove_diacritics(text)
    'Montreal, uber, 12.89, Mere, Francoise, noel, 889, اس, اس'
    """
    nfkd_form = unicodedata.normalize("NFKD", text)
    # unicodedata.combining(char) checks if the character is in
    # composed form (consisting of several unicode chars combined), i.e. a diacritic
    return "".join([char for char in nfkd_form if not unicodedata.combining(char)])


@InputSeries(TextSeries)
def remove_diacritics(s: TextSeries) -> TextSeries:
    """
    Remove all diacritics and accents.

    Remove all diacritics and accents from any word and characters from the
    given Pandas Series.
    Return a cleaned version of the Pandas Series.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(
    ...     "Montréal, über, 12.89, Mère, Françoise, noël, 889, اِس, اُس")
    >>> hero.remove_diacritics(s)[0]
    'Montreal, uber, 12.89, Mere, Francoise, noel, 889, اس, اس'

    """
    return s.astype("unicode").apply(_remove_diacritics)


@InputSeries(TextSeries)
def remove_whitespace(s: TextSeries) -> TextSeries:
    r"""
    Remove any extra white spaces.

    Remove any extra whitespace in the given Pandas Series.
    Remove also newline, tabs and any form of space.

    Useful when there is a need to visualize a Pandas Series and
    most cells have many newlines or other kind of space characters.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Title \n Subtitle \t    ...")
    >>> hero.remove_whitespace(s)
    0    Title Subtitle ...
    dtype: object
    """

    return s.str.replace("\xa0", " ").str.split().str.join(" ")


def _replace_stopwords(text: str, words: Set[str], symbol: str = " ") -> str:
    """
    Remove words in a set from a string, replacing them with a symbol.

    Parameters
    ----------
    text: str

    stopwords : Set[str]
        Set of stopwords string to remove.

    symbol: str, optional, default=" "
        Character(s) to replace words with.

    Examples
    --------
    >>> from texthero.preprocessing import _replace_stopwords
    >>> s = "the book of the jungle"
    >>> symbol = "$"
    >>> stopwords = ["the", "of"]
    >>> _replace_stopwords(s, stopwords, symbol)
    '$ book $ $ jungle'

    """

    pattern = r"""(?x)                          # Set flag to allow verbose regexps
      \w+(?:-\w+)*                              # Words with optional internal hyphens 
      | \s*                                     # Any space
      | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]    # Any symbol 
    """

    return "".join(t if t not in words else symbol for t in re.findall(pattern, text))


@InputSeries(TextSeries)
def replace_stopwords(
    s: TextSeries, symbol: str, stopwords: Optional[Set[str]] = None
) -> TextSeries:
    """
    Replace all instances of `words` with symbol.

    By default uses NLTK's english stopwords of 179 words.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbol: str
        Character(s) to replace words with.

    stopwords : Set[str], optional, default=None
        Set of stopwords string to remove. If not passed,
        by default uses NLTK English stopwords. 

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("the book of the jungle")
    >>> hero.replace_stopwords(s, "X")
    0    X book X X jungle
    dtype: object

    """

    if stopwords is None:
        stopwords = _stopwords.DEFAULT
    return s.apply(_replace_stopwords, args=(stopwords, symbol))


@InputSeries(TextSeries)
def remove_stopwords(
    s: TextSeries, stopwords: Optional[Set[str]] = None, remove_str_numbers=False
) -> TextSeries:
    """
    Remove all instances of `words`.

    By default use NLTK's english stopwords of 179 words:

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    stopwords : Set[str], optional, default=None
        Set of stopwords string to remove. If not passed,
        by default uses NLTK English stopwords.

    Examples
    --------

    Using default NLTK list of stopwords:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero is not only for the heroes")
    >>> hero.remove_stopwords(s)
    0    Texthero      heroes
    dtype: object

    Add custom words into the default list of stopwords:

    >>> import texthero as hero
    >>> from texthero import stopwords
    >>> import pandas as pd
    >>> default_stopwords = stopwords.DEFAULT
    >>> custom_stopwords = default_stopwords.union(set(["heroes"]))
    >>> s = pd.Series("Texthero is not only for the heroes")
    >>> hero.remove_stopwords(s, custom_stopwords)
    0    Texthero      
    dtype: object


    """
    return replace_stopwords(s, symbol="", stopwords=stopwords)


def _replace_emojis(text: str) -> str:
    """
    Replace emojis in a string, replacing them with a placeholder.

    Parameters
    ----------
    text: str

    Examples
    --------
    >>> from texthero.preprocessing import _replace_emojis
    >>> s = "the book of the jungle 😈"
    >>> _replace_emojis(s)
    'the book of the jungle :smiling_face_with_horns:'

    """

    return emoji.demojize(text)


@InputSeries(TextSeries)
def replace_emojis(s: TextSeries) -> TextSeries:
    """
    Replace emojis in a string, replacing them with a placeholder.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("the book of the jungle 😈")
    >>> hero.replace_emojis(s)
    0    the book of the jungle :smiling_face_with_horns:
    dtype: object

    """

    return s.apply(_replace_emojis)


def _place_emojis(text: str) -> str:
    """
    Place back emojis in a string, replacing placeholders with emojis.

    Parameters
    ----------
    text: str

    Examples
    --------
    >>> from texthero.preprocessing import _replace_emojis
    >>> s = "the book of the jungle :smiling_face_with_horns:"
    >>> _place_emojis(s)
    'the book of the jungle 😈'

    """

    return emoji.emojize(text)


@InputSeries(TextSeries)
def place_emojis(s: TextSeries) -> TextSeries:
    """
    Place back emojis in a string, replacing placeholders with emojis.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("the book of the jungle :smiling_face_with_horns:")
    >>> hero.place_emojis(s)
    0    the book of the jungle 😈
    dtype: object

    """

    return s.apply(_place_emojis)


def get_default_pipeline() -> List[Callable[[pd.Series], pd.Series]]:
    """
    Return a list contaning all the methods used in the default cleaning
    pipeline.

    Return a list with the following functions:
     1. :meth:`texthero.preprocessing.fillna`
     2. :meth:`texthero.preprocessing.lowercase`
     3. :meth:`texthero.preprocessing.remove_digits`
     4. :meth:`texthero.preprocessing.remove_punctuation`
     5. :meth:`texthero.preprocessing.remove_diacritics`
     6. :meth:`texthero.preprocessing.remove_stopwords`
     7. :meth:`texthero.preprocessing.remove_whitespace`
    """
    return [
        fillna,
        lowercase,
        remove_digits,
        remove_punctuation,
        remove_diacritics,
        remove_stopwords,
        remove_whitespace,
    ]


@InputSeries(TextSeries)
def clean(s: TextSeries, pipeline=None) -> TextSeries:
    """
    Pre-process a text-based Pandas Series, by using the following default
    pipeline.

     Default pipeline:
     1. :meth:`texthero.preprocessing.fillna`
     2. :meth:`texthero.preprocessing.lowercase`
     3. :meth:`texthero.preprocessing.remove_digits`
     4. :meth:`texthero.preprocessing.remove_punctuation`
     5. :meth:`texthero.preprocessing.remove_diacritics`
     6. :meth:`texthero.preprocessing.remove_stopwords`
     7. :meth:`texthero.preprocessing.remove_whitespace`

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    pipeline : List[Callable[Pandas Series, Pandas Series]],
               optional, default=None
       Specific pipeline to clean the texts. Has to be a list
       of functions taking as input and returning as output
       a Pandas Series. If None, the default pipeline
       is used.
   
    Examples
    --------
    For the default pipeline:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Uper 9dig.        he her ÄÖÜ")
    >>> hero.clean(s)
    0    uper 9dig aou
    dtype: object
    """

    if not pipeline:
        pipeline = get_default_pipeline()

    for f in pipeline:
        s = s.pipe(f)
    return s


@InputSeries(TextSeries)
def has_content(s: TextSeries) -> TextSeries:
    r"""
    Return a Boolean Pandas Series indicating if the rows have content.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["content", np.nan, "\t\n", " "])
    >>> hero.has_content(s)
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    """
    return (s.pipe(remove_whitespace) != "") & (~s.isna())


@InputSeries(TextSeries)
def drop_no_content(s: TextSeries) -> TextSeries:
    r"""
    Drop all rows without content.

    Every row from a given Pandas Series, where :meth:`has_content` is False,
    will be dropped.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["content", np.nan, "\t\n", " "])
    >>> hero.drop_no_content(s)
    0    content
    dtype: object

    """
    return s[has_content(s)]


@InputSeries(TextSeries)
def remove_round_brackets(s: TextSeries) -> TextSeries:
    """
    Remove content within parentheses '()' and the parentheses by themself.

    Examples
    --------

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero (is not a superhero!)")
    >>> hero.remove_round_brackets(s)
    0    Texthero 
    dtype: object

    See also
    --------
    :meth:`remove_brackets`
    :meth:`remove_angle_brackets`
    :meth:`remove_curly_brackets`
    :meth:`remove_square_brackets`

    """
    return s.str.replace(r"\([^()]*\)", "")


@InputSeries(TextSeries)
def remove_curly_brackets(s: TextSeries) -> TextSeries:
    """
    Remove content within curly brackets '{}' and the curly brackets by
    themselves.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero {is not a superhero!}")
    >>> hero.remove_curly_brackets(s)
    0    Texthero 
    dtype: object

    See also
    --------
    :meth:`remove_brackets`
    :meth:`remove_angle_brackets`
    :meth:`remove_round_brackets`
    :meth:`remove_square_brackets`

    """
    return s.str.replace(r"\{[^{}]*\}", "")


@InputSeries(TextSeries)
def remove_square_brackets(s: TextSeries) -> TextSeries:
    """
    Remove content within square brackets '[]' and the square brackets by
    themselves.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero [is not a superhero!]")
    >>> hero.remove_square_brackets(s)
    0    Texthero 
    dtype: object

    See also
    --------
    :meth:`remove_brackets`
    :meth:`remove_angle_brackets`
    :meth:`remove_round_brackets`
    :meth:`remove_curly_brackets`


    """
    return s.str.replace(r"\[[^\[\]]*\]", "")


@InputSeries(TextSeries)
def remove_angle_brackets(s: TextSeries) -> TextSeries:
    """
    Remove content within angle brackets '<>' and the angle brackets by
    themselves.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero <is not a superhero!>")
    >>> hero.remove_angle_brackets(s)
    0    Texthero 
    dtype: object

    See also
    --------
    :meth:`remove_brackets`
    :meth:`remove_round_brackets`
    :meth:`remove_curly_brackets`
    :meth:`remove_square_brackets`

    """
    return s.str.replace(r"<[^<>]*>", "")


@InputSeries(TextSeries)
def remove_brackets(s: TextSeries) -> TextSeries:
    """
    Remove content within brackets and the brackets itself.

    Remove content from any kind of brackets, (), [], {}, <>.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero (round) [square] [curly] [angle]")
    >>> hero.remove_brackets(s)
    0    Texthero    
    dtype: object

    See also
    --------
    :meth:`remove_round_brackets`
    :meth:`remove_curly_brackets`
    :meth:`remove_square_brackets`
    :meth:`remove_angle_brackets`

    """

    return (
        s.pipe(remove_round_brackets)
        .pipe(remove_curly_brackets)
        .pipe(remove_square_brackets)
        .pipe(remove_angle_brackets)
    )


@InputSeries(TextSeries)
def remove_html_tags(s: TextSeries) -> TextSeries:
    """
    Remove html tags from the given Pandas Series.

    Remove all html tags of the type `<.*?>` such as <html>, <p>,
    <div class="hello"> and remove all html tags of type &nbsp and return a
    cleaned Pandas Series.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("<html><h1>Title</h1></html>")
    >>> hero.remove_html_tags(s)
    0    Title
    dtype: object

    """

    pattern = r"""(?x)                              # Turn on free-spacing
      <[^>]+>                                       # Remove <html> tags
      | &([a-z0-9]+|\#[0-9]{1,6}|\#x[0-9a-f]{1,6}); # Remove &nbsp;
      """

    return s.str.replace(pattern, "")


@InputSeries(TextSeries)
def tokenize(s: TextSeries) -> TokenSeries:
    """
    Tokenize each row of the given Series.

    Tokenize each row of the given Pandas Series and return a Pandas Series
    where each row contains a list of tokens.

    Algorithm: add a space between any punctuation symbol at
    exception if the symbol is between two alphanumeric character and split.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Today you're looking great!"])
    >>> hero.tokenize(s)
    0    [Today, you're, looking, great, !]
    dtype: object

    """

    punct = string.punctuation.replace("_", "")
    # In regex, the metacharacter 'w' is "a-z, A-Z, 0-9, including the _ (underscore)
    # character." We therefore remove it from the punctuation string as this is already
    # included in \w.

    pattern = rf"((\w)([{punct}])(?:\B|$)|(?:^|\B)([{punct}])(\w))"

    return s.str.replace(pattern, r"\2 \3 \4 \5").str.split()


# Warning message for not-tokenized inputs
_not_tokenized_warning_message = (
    "It seems like the given Pandas Series s is not tokenized. This function will"
    " tokenize it automatically using hero.tokenize(s) first. You should consider"
    " tokenizing it yourself first with hero.tokenize(s) in the future."
)


def phrases(
    s: TokenSeries, min_count: int = 5, threshold: int = 10, symbol: str = "_"
) -> TokenSeries:
    r"""Group up collocations words

    Given a pandas Series of tokenized strings, group together bigrams where
    each tokens has at least `min_count` term frequency and where the
    `threshold` is larger than the underline formula.

    :math:`\frac{(bigram\_a\_b\_count - min\_count)* len\_vocab }
    { (word\_a\_count * word\_b\_count)}`.

    Parameters
    ----------
    s : :class:`texthero._types.TokenSeries`
    
    min_count : int, optional, default=5
        Ignore tokens with frequency less than this.
        
    threshold : int, optional, default=10
        Ignore tokens with a score under that threshold.
        
    symbol : str, optional, default="_"
        Character used to join collocation words.

    Examples
    --------
    >>> import texthero as hero
    >>> s = pd.Series([['New', 'York', 'is', 'a', 'beautiful', 'city'],
    ...               ['Look', ':', 'New', 'York', '!']])
    >>> hero.phrases(s, min_count=1, threshold=1)
    0    [New_York, is, a, beautiful, city]
    1                [Look, :, New_York, !]
    dtype: object

    Reference
    --------
    `Mikolov, et. al: "Distributed Representations of Words and Phrases and
    their Compositionality"
        <https://arxiv.org/abs/1310.4546>`_

    """

    if not isinstance(s.iloc[0], list):
        warnings.warn(_not_tokenized_warning_message, DeprecationWarning)
        s = tokenize(s)

    delimiter = symbol.encode("utf-8")
    phrases = PhrasesTransformer(
        min_count=min_count, threshold=threshold, delimiter=delimiter
    )
    return pd.Series(phrases.fit_transform(s.values), index=s.index)


@InputSeries(TextSeries)
def replace_urls(s: TextSeries, symbol: str = None) -> TextSeries:
    r"""Replace all urls with the given symbol.

    Replace any urls from the given Pandas Series with the given symbol.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbol : str
        The symbol to which the URL should be changed to.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Go to: https://example.com")
    >>> hero.replace_urls(s, "<URL>")
    0    Go to: <URL>
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.remove_urls`

    """
    pattern = r"http\S+"

    return s.str.replace(pattern, symbol)


@InputSeries(TextSeries)
def remove_urls(s: TextSeries) -> TextSeries:
    r"""Remove all urls from a given Pandas Series.

    Remove all urls and replaces them with a single empty space.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Go to: https://example.com")
    >>> hero.remove_urls(s)
    0    Go to:  
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_urls`

    """

    return replace_urls(s, " ")


@InputSeries(TextSeries)
def replace_tags(s: TextSeries, symbol: str) -> TextSeries:
    """Replace all tags from a given Pandas Series with symbol.

    A tag is a string formed by @ concatenated with a sequence of characters
    and digits. Example: @texthero123.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbols : str
        Symbol to replace tags with.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Hi @texthero123, we will replace you")
    >>> hero.replace_tags(s, symbol='TAG')
    0    Hi TAG, we will replace you
    dtype: object

    """

    pattern = r"@[a-zA-Z0-9]+"
    return s.str.replace(pattern, symbol)


@InputSeries(TextSeries)
def remove_tags(s: TextSeries) -> TextSeries:

    """Remove all tags from a given Pandas Series.

    A tag is a string formed by @ concatenated with a sequence of characters
    and digits. Example: @texthero123. Tags are replaceb by an empty space ` `.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Hi @tag, we will remove you")
    >>> hero.remove_tags(s)
    0    Hi  , we will remove you
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_tags` for replacing a tag with a
        custom symbol.
    """
    return replace_tags(s, " ")


def _encode_string(string: str) -> int:
    string_bytes = string.encode()
    return int.from_bytes(string_bytes, byteorder='big')


def _decode_bytes(integer: int) -> str:
    integer_bytes = integer.to_bytes(((i.bit_length() + 7) // 8), byteorder='big')
    return integer_bytes.decode()


def _add_url_placeholder(url: str) -> str:
    if url in PLACEHOLDERS_DICT:
        return PLACEHOLDERS_DICT[url]
    else:
        code = random.randint(FIRST_INT, LAST_INT)
        while code in PLACEHOLDERS_DICT.values():
            code = random.randint(FIRST_INT, LAST_INT)
        PLACEHOLDERS_DICT[url] = code
    return code


def _add_hashtag_placeholder(hashtag: str) -> str:
    if hashtag in PLACEHOLDERS_DICT:
        return PLACEHOLDERS_DICT[hashtag]
    else:
        code = random.randint(FIRST_INT, LAST_INT)
        while code in PLACEHOLDERS_DICT.values():
            code = random.randint(FIRST_INT, LAST_INT)
        PLACEHOLDERS_DICT[hashtag] = code
    return code


def _add_mention_placeholder(mention: str) -> str:
    if mention in PLACEHOLDERS_DICT:
        return PLACEHOLDERS_DICT[mention]
    else:
        code = random.randint(FIRST_INT, LAST_INT)
        while code in PLACEHOLDERS_DICT.values():
            code = random.randint(FIRST_INT, LAST_INT)
        PLACEHOLDERS_DICT[mention] = code
    return code


@InputSeries(TextSeries)
def replace_hashtags(s: TextSeries, symbol: str = None) -> TextSeries:
    """Replace all hashtags from a Pandas Series with symbol

    A hashtag is a string formed by # concatenated with a sequence of
    characters, digits and underscores. Example: #texthero_123. 

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbol : str
        Symbol to replace hashtags with.
    
    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Hi #texthero_123, we will replace you.")
    >>> hero.replace_hashtags(s, symbol='HASHTAG')
    0    Hi HASHTAG, we will replace you.
    dtype: object

    """
    pattern = r"#[a-zA-Z0-9_]+"
    return s.str.replace(pattern, symbol)


@InputSeries(TextSeries)
def replace_hashtags_w_code(s: TextSeries) -> TextSeries:
    copy = s.copy()
    hashtag_pattern = r"(#[a-zA-Z0-9_]+)"
    hashtags_found_list = copy.str.extractall(hashtag_pattern).reset_index()[0].unique()
    for hashtag in hashtags_found_list:
        copy = copy.str.replace(hashtag, str(_add_hashtag_placeholder(hashtag)))
    return copy


@InputSeries(TextSeries)
def replace_mentions_w_code(s: TextSeries) -> TextSeries:
    copy = s.copy()
    mention_pattern = r"(@[a-zA-Z0-9]+)"
    mentions_found_list = copy.str.extractall(mention_pattern).reset_index()[0].unique()
    for mention in mentions_found_list:
        copy = copy.str.replace(mention, str(_add_mention_placeholder(mention)))
    return copy


@InputSeries(TextSeries)
def replace_urls_w_code(s: TextSeries) -> TextSeries:
    copy = s.copy()
    url_pattern = r"(http\S+)"
    urls_found_list = copy.str.extractall(url_pattern).reset_index()[0].unique()
    for url in urls_found_list:
        copy = copy.str.replace(url, str(_add_url_placeholder(url)))
    return copy


@InputSeries(TextSeries)
def remove_hashtags(s: TextSeries) -> TextSeries:
    """Remove all hashtags from a given Pandas Series

    A hashtag is a string formed by # concatenated with a sequence of
    characters, digits and underscores. Example: #texthero_123. 

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Hi #texthero_123, we will remove you.")
    >>> hero.remove_hashtags(s)
    0    Hi  , we will remove you.
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_hashtags` for replacing a hashtag
        with a custom symbol.
    """
    return replace_hashtags(s, " ")


def _tropical_terms_replacement(text: str) -> str:
    #Tropical terms replacement
    for key, val in tropical_dic.items():
        text = text.lower().replace(key,val)
    return text


def _transfer_casing_for_matching_text(text_w_casing, text_wo_casing):
    """Transferring the casing from one text to another - assuming that
    they are 'matching' texts, alias they have the same length.

    Parameters
    ----------
    text_w_casing : str
        Text with varied casing
    text_wo_casing : str
        Text that is in lowercase only

    Returns
    -------
    str
        Text with the content of `text_wo_casing` and the casing of
        `text_w_casing`

    Raises
    ------
    ValueError
        If the input texts have different lengths
    """
    if len(text_w_casing) != len(text_wo_casing):
        raise ValueError("The 'text_w_casing' and 'text_wo_casing' "
                         "don't have the same length, "
                         "so you can't use them with this method, "
                         "you should be using the more general "
                         "transfer_casing_similar_text() method.")

    return ''.join([y.upper() if x.isupper() else y.lower()
                    for x, y in zip(text_w_casing, text_wo_casing)])


def _transfer_casing_for_similar_text(text_w_casing, text_wo_casing):
    """Transferring the casing from one text to another - for similar
    (not matching) text

    1. It will use `difflib`'s `SequenceMatcher` to identify the
       different type of changes needed to turn `text_w_casing` into
       `text_wo_casing`
    2. For each type of change:

       - for inserted sections:

         - it will transfer the casing from the prior character
         - if no character before or the character before is the\
           space, then it will transfer the casing from the following\
           character

       - for deleted sections: no case transfer is required
       - for equal sections: just swap out the text with the original,\
         the one with the casings, as otherwise the two are the same
       - replaced sections: transfer the casing using\
         :meth:`transfer_casing_for_matching_text` if the two has the\
         same length, otherwise transfer character-by-character and\
         carry the last casing over to any additional characters.

    Parameters
    ----------
    text_w_casing : str
        Text with varied casing
    text_wo_casing : str
        Text that is in lowercase only

    Returns
    -------
    text_wo_casing : str
        If `text_wo_casing` is empty
    c : str
        Text with the content of `text_wo_casing` but the casing of
        `text_w_casing`

    Raises
    ------
    ValueError
        If `text_w_casing` is empty
    """
    if not text_wo_casing:
        return text_wo_casing

    if not text_w_casing:
        raise ValueError("We need 'text_w_casing' to know what "
                         "casing to transfer!")

    _sm = SequenceMatcher(None, text_w_casing.lower(),
                          text_wo_casing)

    # we will collect the case_text:
    c = ''

    # get the operation codes describing the differences between the
    # two strings and handle them based on the per operation code rules
    for tag, i1, i2, j1, j2 in _sm.get_opcodes():
        # Print the operation codes from the SequenceMatcher:
        # print("{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}"
        #       .format(tag, i1, i2, j1, j2,
        #               text_w_casing[i1:i2],
        #               text_wo_casing[j1:j2]))

        # inserted character(s)
        if tag == "insert":
            # if this is the first character and so there is no
            # character on the left of this or the left of it a space
            # then take the casing from the following character
            if i1 == 0 or text_w_casing[i1 - 1] == " ":
                if text_w_casing[i1] and text_w_casing[i1].isupper():
                    c += text_wo_casing[j1 : j2].upper()
                else:
                    c += text_wo_casing[j1 : j2].lower()
            else:
                # otherwise just take the casing from the prior
                # character
                if text_w_casing[i1 - 1].isupper():
                    c += text_wo_casing[j1 : j2].upper()
                else:
                    c += text_wo_casing[j1 : j2].lower()

        elif tag == "delete":
            # for deleted characters we don't need to do anything
            pass

        elif tag == "equal":
            # for 'equal' we just transfer the text from the
            # text_w_casing, as anyhow they are equal (without the
            # casing)
            c += text_w_casing[i1 : i2]

        elif tag == "replace":
            _w_casing = text_w_casing[i1 : i2]
            _wo_casing = text_wo_casing[j1 : j2]

            # if they are the same length, the transfer is easy
            if len(_w_casing) == len(_wo_casing):
                c += _transfer_casing_for_matching_text(
                    text_w_casing=_w_casing, text_wo_casing=_wo_casing)
            else:
                # if the replaced has a different length, then we
                # transfer the casing character-by-character and using
                # the last casing to continue if we run out of the
                # sequence
                _last = "lower"
                for w, wo in zip_longest(_w_casing, _wo_casing):
                    if w and wo:
                        if w.isupper():
                            c += wo.upper()
                            _last = "upper"
                        else:
                            c += wo.lower()
                            _last = "lower"
                    elif not w and wo:
                        # once we ran out of 'w', we will carry over
                        # the last casing to any additional 'wo'
                        # characters
                        c += wo.upper() if _last == "upper" else wo.lower()
    return c    


def _check_spelling(text: str) -> str:
    """
    Check Spanish spelling of a string, replacing mispelled words and slang words in tropical_dict.json glossary

    Parameters
    ----------
    text: str

    Examples
    --------
    >>> from texthero.preprocessing import _check_spelling
    >>> s = "the book of the jungle :smiling_face_with_horns:"
    >>> _place_emojis(s)
    'the book of the jungle 😈'

    """
    text_w_casing = text
    text_wo_casing = _tropical_terms_replacement(text)
    text = _transfer_casing_for_similar_text(text_w_casing, text_wo_casing)
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2, ignore_non_words=False, transfer_casing=True, split_phrase_by_space=False)
    best_suggestion = str(suggestions[0])[:-6].replace(' .','.').replace(' ,',',')
    return best_suggestion


@InputSeries(TextSeries)
def check_spelling(s: TextSeries) -> TextSeries:
    return s.apply(_check_spelling)


def get_twitter_pipeline() -> List[Callable[[pd.Series], pd.Series]]:
    """
    Return a list contaning all the methods used in the default cleaning
    pipeline.

    Return a list with the following functions:
     1. :meth:`texthero.preprocessing.fillna`
     2. :meth:`texthero.preprocessing.replace_emojis`
     3. :meth:`texthero.preprocessing.replace_hashtags`
     4. :meth:`texthero.preprocessing.replace_urls`
     5. :meth:`texthero.preprocessing.remove_whitespace`
    """
    return [
        fillna,
        replace_emojis,
        replace_urls_w_code,
        replace_hashtags_w_code,
        replace_mentions_w_code,
        check_spelling,
        remove_whitespace
    ]


@InputSeries(TextSeries)
def clean_tweets(s: TextSeries, pipeline=get_twitter_pipeline()) -> TextSeries:
    """
    Pre-process a text-based Pandas Series of tweets, by using the following
    pipeline.

     Twitter pipeline:
     1. :meth:`texthero.preprocessing.fillna`
     2. :meth:`texthero.preprocessing.replace_emojis`
     3. :meth:`texthero.preprocessing.replace_urls`

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    pipeline : List[Callable[Pandas Series, Pandas Series]],
               optional, default=None
       Specific pipeline to clean the texts. Has to be a list
       of functions taking as input and returning as output
       a Pandas Series. If None, the default pipeline
       is used.
   
    Examples
    --------
    For the default pipeline:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("the book of the jungle 😈 https://example.com")
    >>> hero.clean_tweets(s)
    0    the book of the jungle :smiling_face_with_horns: <URL>
    dtype: object
    """

    if not pipeline:
        pipeline = get_twitter_pipeline()

    for f in pipeline:
        s = s.pipe(f)
    return s
