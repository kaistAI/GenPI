pseudo_input_system_prompt = """
The document below are the code of conduct for a specific agent performing the given problem.

### This is the given Document:
-----------
{context}
-----------
""".lstrip()



pseudo_input_user_prompt = """
Based on the given document, generate several problems for Natural Language to Bash script task.
Generate only problems with a numbered list.
This is {num_gen_once} generated Problems.

### Problems:
{inputs}
""".strip()



# pseudo_student_input_prompt = """
# Generate answer from the given problem.
# Generate the answer by wrapping it in bash markdown format.

# ### Problem:
# {input}

# ### Answer:
# ```bash
# """.lstrip()
pseudo_student_input_prompt = """
I will start a new problem in a new OS. My problem is:

{input}
""".lstrip()



pseudo_teacher_input_prompt = """
{context}

{input}
""".lstrip()



reason_system_prompt = """
You are a commentator comparing the answers of two models to a question.
I will give you four components: context, input, bad output, good output.
Bad Output is a response provided without any context, only based on the input given.
Good Output is a response provided with both context and input.
Explain the reason for the change in response, referring to the context, in about five sentences.
""".strip()



reason_user_prompt = """
### Context:
-----------
{context}
-----------

### Input:
{input}

### Bad Output:
{student_output}

### Good Output:
{teacher_output}

### Reason:
""".lstrip()



meta_cognition_input_prompt = """
When a certain Context and Input were provided together, the Bad Output changed to Good Output as shown below.
Guess which Context was provided and generate a Reason why the Output changed as shown below.

### Input:
{input}

### Bad Output:
{student_output}

### Good Output:
{teacher_output}
""".strip()



meta_cognition_output_prompt = """
### Context
-----------
{context}
-----------

### Reason
{reason}
""".strip()