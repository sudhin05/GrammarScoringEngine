from transformers import pipeline

model_id = "google/flan-t5-base"

pipe = pipeline(
    "text2text-generation",
    model=model_id,
    device="cuda:0"
)

prompt = "Judge the grammatical correctness of the input audio sentence and classify as" \
"1 : The person's speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns."\
"2 : The person has a limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic sentence structure and grammatical mistakes. They might leave sentences incomplete."\
"3 : The person demonstrates a decent grasp of sentence structure but makes errors in grammatical structure, or they show a decent grasp of grammatical structure but make errors in sentence syntax and structure."\
"4 : The person displays a strong understanding of sentence structure and syntax. They consistently show good control of grammar. While occasional errors may occur, they are generally minor and do not lead to misunderstandings; the person can correct most of them."\
"5 : Overall, the person showcases high grammatical accuracy and adept control of complex grammar. They use grammar accurately and effectively, seldom making noticeable mistakes. Additionally, they handle complex language structures well and correct themselves when necessary."

statement = "In the background of a school, you might see bustling hallways filled with students hurrying to their next class. Colorful bulletin boards adorned with students' art work and academic achievements. The teachers engage in animated discussions with their students."
input_sent = prompt + statement

prediction = pipe(input_sent)

print(prediction)