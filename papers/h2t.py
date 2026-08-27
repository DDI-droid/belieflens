import re,sys,html
def conv(p,o):
    s=open(p,encoding='utf-8',errors='ignore').read()
    s=re.sub(r'(?is)<(script|style|head)\b.*?</\1>',' ',s)
    s=re.sub(r'(?is)<math\b[^>]*alttext="([^"]*)"[^>]*>.*?</math>',lambda m:' $'+html.unescape(m.group(1))+'$ ',s)
    s=re.sub(r'(?is)<math\b.*?</math>',' [math] ',s)
    s=re.sub(r'(?is)</(p|div|li|tr|h1|h2|h3|h4|section|figure|table|blockquote)>','\n',s)
    s=re.sub(r'(?is)<br\s*/?>','\n',s)
    s=re.sub(r'(?is)<td\b[^>]*>',' | ',s)
    s=re.sub(r'(?s)<[^>]+>',' ',s)
    s=html.unescape(s)
    s=re.sub(r'[ \t\xa0]+',' ',s)
    s=re.sub(r'\n\s*\n+','\n\n',s)
    open(o,'w',encoding='utf-8').write(s.strip())
    print(o, len(s))
for a,b in [('full_2604.23072.html','analytica.txt'),('full_2605.15188.html','futuresim.txt'),('full_2605.11436.html','agentbrace.txt')]:
    conv(a,b)
