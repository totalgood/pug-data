# from future import print_function, absolute_import
import gzip
import re

PATH_SHAKESPEARE = 'cache/shakespeare-complete-works.txt.gz'

RE_TITLE = re.compile(r'^(([-\'A-Z]+[ ]?){3,8})$')
RE_YEAR = re.compile(r'^1[56][0-9]{2}$')
RE_THE_END = re.compile(r'^THE[ ]END$')
RE_BYLINE = re.compile(r'^by William Shakespeare$')


books = [{}]
books[0]['title'] = 'THE SONNETS'
books[0]['start'] = 172
books[0]['stop'] = 2797


def generate_lines(path=PATH_SHAKESPEARE,
                   start=0,
                   stop=float('inf')):
    """Generate (yield) lines in a gzipped file (*.txt.gz) one line at a time"""
    with (gzip.open(path, 'rU') if path.endswith('.gz') else open(path, 'rU')) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= stop:
                break
            yield line.rstrip()


def segment_books(path=PATH_SHAKESPEARE, books=books):
    j, k = 0, 0
    for i, line in enumerate(generate_lines()):
        if j >= len(books):
            books += [{}]
        if not len(books[j]):
            match = RE_YEAR.match(line)
            if match:
                print(" year {:02d}, {}: {}".format(j, k, match.group()))
                k = 1
                books[j]['year'] = int(match.group())
                books[j]['start'] = i
        else:
            if 'start' in books[j]:
                match = RE_TITLE.match(line)
                if match:
                    print("title {:02d}, {}: {}".format(j, k, match.groups()[0]))
                    k = 3
                    books[j]['title'] = match.groups()[0]
            if 'title' in books[j]:
                match = RE_BYLINE.match(line)
                if match:
                    print("   by {:02d}, {}: {}".format(j, k, match.group()))
                    k = 2
                    books[j]['by'] = match.group()[3:]
            if 'title' in books[j]:
                match = RE_THE_END.match(line)
                if match and 'GUTENBERG' not in match.group().upper():
                    print(" stop {:02d}, {}: {}".format(j, k, match.group()))
                    books[j]['stop'] = i
                    j += 1
                    k = 0
    return books
