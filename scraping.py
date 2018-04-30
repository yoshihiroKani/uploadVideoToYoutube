#coding: UTF-8
from readability.readability import Document
import urllib

url = 'https://headlines.yahoo.co.jp/hl?a=20180430-00000098-jij-spo'

html = urllib.urlopen(url).read()
readable_article = Document(html).summary()
readable_title = Document(html).short_title()

print readable_article
print readable_title
