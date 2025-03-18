from llamaInstruct3 import llama3Instruct3
from llama3 import llama3

with open("outputs_llama3.txt", "w") as wfile:
    with open('inputs.txt', 'r') as file:
        for line in file.readlines():
            llama3_obj = llama3()
            wfile.write(llama3_obj.predict(line) + '\n')



# with open("outputs_llamaInstruct3.txt", "w") as wfile:
#     with open('inputs.txt', 'r') as file:
#         for line in file.readlines():
#             llama3_obj = llama3Instruct3()
#             wfile.write(llama3_obj.predict(line) + '\n')