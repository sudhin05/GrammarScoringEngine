import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def check_grammar(text):
    matches = tool.check(text)
    return matches

def grammar_score(text):
    words = len(text.split())
    errors = len(check_grammar(text))
    print(int((errors / max(words, 1)) * 100))
    score = max(0, 100 - int((errors / max(words, 1)) * 100))
    return score

transcription="Uh, let's see, I'm at Eastern Market. Um, the place is crowded. They're selling everything from clothing to jewelry to shoes. Um, there's street vendors here and they're selling hot dogs, hamburgers, sliders, fish sandwiches, Mexican food. Chinese food. It's pretty loud here so I have to like talk over the mic and  in the morning it's a pretty relaxed crowd mostly old people looking for some  great deals and then as it transitions into the evening you get more of the  urban crowd. single moms kids teenagers and then by night you get the rowdy a bunch the people coming home from  the bar um who just want to eat some of this delicious street food i don't know i would say  that the eastern market is a very good market you should come here when you get the chance"

score = grammar_score(transcription)
print(f"✅ Grammar Score: {score}/100\n")