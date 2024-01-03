LangChain (python and JavaScript/Typescript)
#!pip install --upgrade langchain

- Models
- Prompts: e.g., Though, Action, Observation as keywords for Chain-of-Thought Reasoning (ReAct)
- Memory
  * ConversationBufferMemory (<span style="color:blue">see codes</span>)
  * ConversationBufferWindowMemory(k=1) [k=1 remembers last conversation]
  * ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
  * ConversationSummaryMemory(llm=llm, max_token_limit=400) - write summary and use as memory
    Also see vector data memory - retrieves relevant conversation
- Indexes
- Chains
  * LLMChain
  * Sequential Chains
    * SimpleSequentialChain - one input and one output
    * SequentialChain - output of chains 1 and 2 as input for chain 3
  * Router Chain - decides which subchain to pass to
- Agents


Codes:

```
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
conversation.predict(input="")
print(memory.buffer)
memory.load_memory_variables({})
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})

review_template = "What is the best name to describe \
    a company that makes {product}?"
prompt_template = ChatPromptTemplate.from_template(review_template)

chain = LLMChain(llm=llm, prompt=prompt)  # chain is combination of llm and prompt
chain.run(product)

first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
                    )
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)
overall_chain(df.Review[5])

prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )
chain.run("What is black body radiation?")
```

# RAG

```
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator

# Option 1: Few-liner
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
response = index.query(query)

# Option 2: Step by Step (with more control)
from langchain.embeddings import OpenAIEmbeddings
loader = CSVLoader(file_path=file)
docs = loader.load()  # check: docs[0]
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query(query)
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
docs = db.similarity_search(query)
retriever = db.as_retriever()
llm = ChatOpenAI(temperature = 0.0, model=llm_model)

### Option 2a: Doing it manually ###
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")
display(Markdown(response))

### Option 2b: Through chain ###
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", # Map_reduce, Refine, Map_rerank
    retriever=retriever,   # or index.vectorestore.as_retriever()
    verbose=True,
    # chain_type_kwargs = "document_separator":"<<<<>>>>"
)
response = qa_stuff.run(query)  
display(Markdown(response))

### Option 2c: One-liner ### 
response = index.query(query, llm=llm)
display(Markdown(response))

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])
```

### LLM-Generated examples

```
from langchain.evaluation.qa import QAGenerateChain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
import langchain
langchain.debug = False

predictions = qa.apply(examples)
from langchain.evaluation.qa import QAEvalChain
llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
```

### Agents
```
tools = load_tools(["llm-math","wikipedia"], llm=llm)
agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
agent(query)

agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True

agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
# connect to own API, data
)

```