# from future import print_function, absolute_import
import gzip
import re
import os

from pug.data.constant import DATA_PATH, DICT_ROMAN2INT


PATH_SHAKESPEARE = os.path.join(DATA_PATH, 'shakespeare-complete-works.txt.gz')

RE_TITLE = re.compile(r'(([-;,\'A-Z]+[ ]?){3,8})')
RE_TITLE_LINE = re.compile(r'^' + RE_TITLE.pattern + r'$')
RE_GUTEN_LINE = re.compile(r'^\*\*\*\ START\ OF\ THIS\ PROJECT\ GUTENBERG\ EBOOK\ ' +
                           RE_TITLE.pattern + r'\ \*\*\*$')
RE_ACT_SCENE_LINE = re.compile(r'^((ACT\ [IV]+)[.]?\ (SCENE\ [0-9]{1,2})[.]?)$')
RE_YEAR_LINE = re.compile(r'^1[56][0-9]{2}$')
RE_THE_END = re.compile(r'^THE[ ]END$')
RE_BY_LINE = re.compile(r'^((by\ )?(William\ Shakespeare))$', re.IGNORECASE)


def generate_lines(path=PATH_SHAKESPEARE,
                   start=0,
                   stop=float('inf')):
    r"""Generate (yield) lines in a gzipped file (*.txt.gz) one line at a time

    >>> i = 0
    >>> for i, line in enumerate(generate_lines()):
    ...     pass
    >>> i
    124786
    >>> iter(generate_lines()).next()
    '\xef\xbb\xbfThe Project Gutenberg EBook of The Complete Works of William Shakespeare, by'
    """
    with (gzip.open(path, 'rU') if path.endswith('.gz') else open(path, 'rU')) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= stop:
                break
            yield line.rstrip()


def segment_shakespeare_works(path=PATH_SHAKESPEARE, verbose=False):
    """Find start and end of each volume within _Complete Works of William Shakespeare_

    >>> segment_shakespeare_works()
    {'body_start': 21,
     'title': 'COMPLETE WORKS--WILLIAM SHAKESPEARE',
     'volumes': [{'by': 'William Shakespeare',
       'by_lineno': 177,
       'start': 173,
       'stop': 2798,
       'title': 'THE SONNETS',
       'title_lineno': 175,
       'year': 1609},
    ...
    {'by': 'William Shakespeare',
     'by_lineno': 123986,
     'start': 123982,
     'stop': 124366,
     'title': "A LOVER'S COMPLAINT",
     'title_lineno': 123984,
     'year': 1609}]}
    """
    works = [{}]
    meta = {}
    j = 0
    for i, line in enumerate(generate_lines(path=path)):
        if 'title' not in meta:
            match = RE_GUTEN_LINE.match(line)
            if match:
                meta['title'] = match.groups()[0]
                meta['body_start'] = i
            continue
        if j >= len(works):
            works += [{}]
        if not len(works[j]):
            match = RE_YEAR_LINE.match(line)
            if match:
                if verbose:
                    print(" year {:02d}, {}: {}".format(j, i, match.group()))
                works[j]['year'] = int(match.group())
                works[j]['start'] = i
        elif len(works[j]) == 2:
            match = RE_TITLE_LINE.match(line)
            if match:
                if verbose:
                    print("title {:02d}, {}: {}".format(j, i, match.groups()[0]))
                works[j]['title'] = match.groups()[0]
                works[j]['title_lineno'] = i
        elif len(works[j]) == 4:
            match = RE_BY_LINE.match(line)
            if match:
                if verbose:
                    print("   by {:02d}, {}: {}".format(j, i, match.group()))
                works[j]['by'] = match.groups()[2]
                works[j]['by_lineno'] = i
        elif len(works[j]) > 4:
            match = RE_ACT_SCENE_LINE.match(line)
            if match:
                section_meta = {
                    'start': i,
                    'title': match.groups()[0],
                    'act_roman': match.groups()[1].split()[-1],
                    'act': int(DICT_ROMAN2INT[match.groups()[1].split()[-1]]),
                    'scene': int(match.groups()[2].split()[-1]),
                }
                works[j]['sections'] = works[j].get('sections', []) + [section_meta]
            else:
                match = RE_THE_END.match(line)
                if match and 'GUTENBERG' not in match.group().upper():
                    if verbose:
                        print(" stop {:02d}, {}: {}".format(j, i, match.group()))
                    works[j]['stop'] = i
                    j += 1
    if not len(works[-1]):
        works = works[:-1]
    meta['volumes'] = works
    return meta
