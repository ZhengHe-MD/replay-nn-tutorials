import os.path
import re
import urllib.request

from bs4 import BeautifulSoup
from bs4.element import Comment
from bs4.element import PageElement


def get_all_links():
    seed = expand_href("articles.html")
    soup = cook(seed)
    tags = soup.findAll('a')

    all_links = set([])
    for tag in tags:
        href = tag.get('href')
        if "html" not in href:
            continue
        if href == "index.html" or href == "rss.html":
            continue
        all_links.add(expand_href(href))
    return list(all_links)


def parse_one_page(blog_link: str):
    body = urllib.request.urlopen(blog_link).read()
    soup = BeautifulSoup(body, 'html.parser')
    text_nodes = soup.findAll(text=True)
    visible_texts = []
    last_text_node = None
    for text_node in text_nodes:
        # ignore invisible texts, intentionally keep the title
        if text_node.parent.name in ['style', 'script', 'head', 'meta', '[document]']:
            continue
        # ignore comments
        if isinstance(text_node, Comment):
            continue
        # remove empty lines
        if len(text_node.text.strip()) == 0:
            continue

        if should_merge(text_node) or should_merge(last_text_node):
            visible_texts[-1] = clean_up_line(' '.join([visible_texts[-1], clean_up_line(text_node.text)]))
            last_text_node = text_node
            continue
        last_text_node = text_node
        visible_texts.append(clean_up_line(text_node.text))
    return "\n".join(text.strip() for text in visible_texts)


def should_merge(text_node: PageElement):
    if text_node is None:
        return False
    font_parent_node = text_node.find_parent('font')
    return text_node.find_parent('a') or \
           text_node.find_parent('b') or \
           text_node.find_parent('i') or \
           (font_parent_node and font_parent_node.has_attr('color'))


def clean_up_line(s: str):
    s = s.replace('\n', ' ')
    s = s.replace(' , ', ', ')
    s = s.replace(' . ', '. ')
    s = s.replace(' ; ', '; ')
    s = s.replace(' ! ', '! ')
    s = re.sub(r'\s+', ' ', s)
    return s


def cook(link: str):
    body = urllib.request.urlopen(link).read()
    soup = BeautifulSoup(body, 'html.parser')
    return soup


domain = "http://www.paulgraham.com/"


def expand_href(href: str):
    return f'{domain}{href}'


def output_name_from_link(link: str):
    basename = os.path.basename(link)
    return '.'.join([basename.split('.')[0], 'txt'])


if __name__ == '__main__':
    links = get_all_links()
    for i, link in enumerate(links):
        print(f'processing {link}')
        page_text = parse_one_page(link)
        with open(output_name_from_link(link), 'w') as f:
            f.write(page_text)
