from pathlib import Path
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict
import random
random.seed(42)

from src.utils import get_project_root


class AgentBenchDataset:
    def __init__(self, environment_name='os', dataset_mode='original', load_dataset_path=''):
        if environment_name=='os':
            from src.dataset_cls.agentbench.os.prompts import pseudo_input_system_prompt, pseudo_input_user_prompt # pseudo-input
            from src.dataset_cls.agentbench.os.prompts import pseudo_student_input_prompt, pseudo_teacher_input_prompt # outputs
            from src.dataset_cls.agentbench.os.prompts import reason_system_prompt, reason_user_prompt # reason
            from src.dataset_cls.agentbench.os.prompts import meta_cognition_input_prompt, meta_cognition_output_prompt  # stage1

            self.pseudo_input_system_prompt = pseudo_input_system_prompt
            self.pseudo_input_user_prompt = pseudo_input_user_prompt
            self.pseudo_student_input_prompt = pseudo_student_input_prompt
            self.pseudo_teacher_input_prompt = pseudo_teacher_input_prompt
            self.reason_system_prompt = reason_system_prompt
            self.reason_user_prompt = reason_user_prompt
            self.meta_cognition_input_prompt = meta_cognition_input_prompt
            self.meta_cognition_output_prompt = meta_cognition_output_prompt

            with open(Path(get_project_root(), 'src/dataset_cls/agentbench/os/agent_prompt.txt')) as f:
                self.agent_prompt = ''.join(f.readlines())
            self.pseudo_input_demons_list = [
                "Tell me whether npm is installed or not. If so, return 'installed'. If not, return 'not-yet'",
                "How many empty files are there in /home and all of its content?",
                "Find out how many groups with an index greater than 50.",
                "Tell me the max number of threads in my computer.",
                "Stock logs are shown in /usr/stock.log. The last two columns are stock index and count respectively. Tell me how many times Bob sold a stock."
            ]
        elif environment_name=='kg':
            from src.dataset_cls.agentbench.kg.prompts import pseudo_input_system_prompt, pseudo_input_user_prompt # pseudo-input
            from src.dataset_cls.agentbench.kg.prompts import pseudo_student_input_prompt, pseudo_teacher_input_prompt # outputs
            from src.dataset_cls.agentbench.kg.prompts import reason_system_prompt, reason_user_prompt # reason
            from src.dataset_cls.agentbench.kg.prompts import meta_cognition_input_prompt, meta_cognition_output_prompt  # stage1

            self.pseudo_input_system_prompt = pseudo_input_system_prompt
            self.pseudo_input_user_prompt = pseudo_input_user_prompt
            self.pseudo_student_input_prompt = pseudo_student_input_prompt
            self.pseudo_teacher_input_prompt = pseudo_teacher_input_prompt
            self.reason_system_prompt = reason_system_prompt
            self.reason_user_prompt = reason_user_prompt
            self.meta_cognition_input_prompt = meta_cognition_input_prompt
            self.meta_cognition_output_prompt = meta_cognition_output_prompt

            with open(Path(get_project_root(), 'src/dataset_cls/agentbench/kg/agent_prompt.txt')) as f:
                self.agent_prompt = ''.join(f.readlines())
            self.pseudo_input_demons_list = [
                "Question: how many mac models used motorola 68040 processors?<SEP>Entities: [mac, Motorola 68040]",
                "Question: identify the tropical cyclones that are in the same category with hurricane marie and also affected eastern north america.<SEP>Entities: [Hurricane Marie, Eastern North America]",
                "Question: which martial art has the same category as silat and has internal?<SEP>Entities: [Silat, Internal]",
                "Question: What is the predominant language of the region where \"Into the Arms of Strangers: Stories of the Kindertransport\" was located?<SEP>Entities: [Into the Arms of Strangers: Stories of the Kindertransport]",
                "Question: how many electronic arts games are available for purchase in the united states of america?<SEP>Entities: [Electronic Arts, United States of America]"
            ]
        elif environment_name=='m2w':
            from src.dataset_cls.agentbench.m2w.prompts import pseudo_input_system_prompt, pseudo_input_user_prompt # pseudo-input
            from src.dataset_cls.agentbench.m2w.prompts import pseudo_student_input_prompt, pseudo_teacher_input_prompt # outputs
            from src.dataset_cls.agentbench.m2w.prompts import reason_system_prompt, reason_user_prompt # reason
            from src.dataset_cls.agentbench.m2w.prompts import meta_cognition_input_prompt, meta_cognition_output_prompt  # stage1

            self.pseudo_input_system_prompt = pseudo_input_system_prompt
            self.pseudo_input_user_prompt = pseudo_input_user_prompt
            self.pseudo_student_input_prompt = pseudo_student_input_prompt
            self.pseudo_teacher_input_prompt = pseudo_teacher_input_prompt
            self.reason_system_prompt = reason_system_prompt
            self.reason_user_prompt = reason_user_prompt
            self.meta_cognition_input_prompt = meta_cognition_input_prompt
            self.meta_cognition_output_prompt = meta_cognition_output_prompt

            with open(Path(get_project_root(), 'src/dataset_cls/agentbench/m2w/agent_prompt.txt')) as f:
                self.agent_prompt = ''.join(f.readlines())
            self.pseudo_input_demons_list = [
                "'''\n<html> <ul> <li> <a grocery clerk jobs> Grocery Clerk jobs </a> <a id=0> Select City </a> </li> <li> <a grocery store jobs> Grocery Store jobs </a> <a id=1> Select City </a> </li> </ul> </html>\n'''\n\nBased on the HTML webpage above, try to complete the following task:\nTask: Find a grocery store cashier job in Florida.\nPrevious actions:\n[link]  Browse Jobs -> CLICK\n[link]  Retail -> CLICK\nWhat should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):\n\nA. None of the above\nB. <a id=0> Select City </a>\nC. <a id=1> Select City </a>\n",
                "'''\n<html> <div> <ul> <a id=0> Baby </a> <a id=1> Child </a> </ul> <div> <a id=2> <div> <img babycenter's picks for best baby /> <div> <span> Best baby sleep sacks </span> <div> Reviewed by Sarah Gard Lazarus, D.O., pediatric emergency medicine physician </div> </div> </div> </a> <img id=3 image contains babycenter best crib /> <img id=4 p'kolino toddler's bed /> </div> </div> </html>\n'''\n\nBased on the HTML webpage above, try to complete the following task:\nTask: check out the best beds available for toddlers\nPrevious actions:\n[link]  BABY PRODUCTS -> CLICK\n[link]  see all sleep -> CLICK\n[button]  Show more -> CLICK\n[button]  Show more -> CLICK\nWhat should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):\n\nA. None of the above\nB. <a id=0> Baby </a>\nC. <a id=1> Child </a>\nD. <a id=2> <div> <img babycenter's picks for best baby />\nE. <img id=3 image contains babycenter best crib />\nF. <img id=4 p'kolino toddler's bed />\n",
                "'''\n<html> <div> <a id=0> <strong> #1 </strong> <div> Princeton University </div> </a> <div> <a> Log In to Compass </a> <a> My Fit Custom Ranking </a> <a id=1> My Schools </a> <a> My Scholarships </a> <a> My Notes </a> <a> Unlock more with Compass </a> </div> </div> </html>\n'''\n\nBased on the HTML webpage above, try to complete the following task:\nTask: Add Princeton University to a list of favorite schools\nPrevious actions:\nNone\nWhat should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):\n\nA. None of the above\nB. <a id=0> <strong> #1 </strong> <div> Princeton University </div> </a>\nC. <a id=1> My Schools </a>\n",
                "'''\n<html> <div> <a id=0> <div> SAP S/4HANA </div> </a> <a> How to Lease Option a Business <div> <span> 10 reviews </span> <span> 1 total hour </span> <span> 8 lectures </span> <span id=1> Intermediate </span> <span> Current price: $19.99 </span> </div> </a> </div> </html>\n'''\n\nBased on the HTML webpage above, try to complete the following task:\nTask: Get the highest rated SAP S/4 HANA course rated 4, and up with a duration between 3 to 6 hours for an intermediate, and add this to your cart and checkout.\nPrevious actions:\n[button]  Categories -> CLICK\n[link]  Design -> HOVER\n[link]  SAP -> HOVER\nWhat should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):\n\nA. None of the above\nB. <a id=0> <div> SAP S/4HANA </div> </a>\nC. <span id=1> Intermediate </span>\n",
                "'''\n<html> <div> <nav specialty forecasts> <a id=0> <span> Cold & Flu </span> </a> </nav> <main main content> <section all pollutants> <button button close> <svg id=1 close> <title> Close </title> </svg> </button> </section> </main> </div> </html>\n'''\n\nBased on the HTML webpage above, try to complete the following task:\nTask: Find out the cold and flu forecast and today's air quality in Champaign, IL.\nPrevious actions:\n[combobox]  Search City or Zip Code -> TYPE: Champaign, IL\n[option]  Champaign, IL -> CLICK\n[button]  More Forecasts Arrow down -> CLICK\n[link]  Air Quality Forecast -> CLICK\n[img]  Arrow down -> CLICK\nWhat should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):\n\nA. None of the above\nB. <a id=0> <span> Cold & Flu </span> </a>\nC. <svg id=1 close> <title> Close </title> </svg>\n"
            ]
        elif environment_name=='webshop':
            from src.dataset_cls.agentbench.webshop.prompts import pseudo_input_system_prompt, pseudo_input_user_prompt # pseudo-input
            from src.dataset_cls.agentbench.webshop.prompts import pseudo_student_input_prompt, pseudo_teacher_input_prompt # outputs
            from src.dataset_cls.agentbench.webshop.prompts import reason_system_prompt, reason_user_prompt # reason
            from src.dataset_cls.agentbench.webshop.prompts import meta_cognition_input_prompt, meta_cognition_output_prompt  # stage1

            self.pseudo_input_system_prompt = pseudo_input_system_prompt
            self.pseudo_input_user_prompt = pseudo_input_user_prompt
            self.pseudo_student_input_prompt = pseudo_student_input_prompt
            self.pseudo_teacher_input_prompt = pseudo_teacher_input_prompt
            self.reason_system_prompt = reason_system_prompt
            self.reason_user_prompt = reason_user_prompt
            self.meta_cognition_input_prompt = meta_cognition_input_prompt
            self.meta_cognition_output_prompt = meta_cognition_output_prompt

            with open(Path(get_project_root(), 'src/dataset_cls/agentbench/webshop/agent_prompt.txt')) as f:
                self.agent_prompt = ''.join(f.readlines())
            self.pseudo_input_demons_list = [
                "Observation:\nWebShop [SEP] Instruction: [SEP] i am looking for a 12 ounce jar of raspberry preserve that is nut and gluten free, and price lower than 130.00 dollars [SEP] Search\n\nAvailable Actions:\n{'has_search_bar': True, 'clickables': ['search']}",
                "Observation:\nWebShop [SEP] Instruction: [SEP] i'm looking for some gluten free jelly with black sesames, and price lower than 30.00 dollars [SEP] Search\n\nAvailable Actions:\n{'has_search_bar': True, 'clickables': ['search']}",
                "Observation:\nWebShop [SEP] Instruction: [SEP] i would like a solid wood sideboard, and price lower than 980.00 dollars [SEP] Search\n\nAvailable Actions:\n{'has_search_bar': True, 'clickables': ['search']}",
                "Observation:\nWebShop [SEP] Instruction: [SEP] i’m looking for a large multi-pack of sweetener that contains no sugar; please pick the blue raspberry flavour, and price lower than 60.00 dollars [SEP] Search\n\nAvailable Actions:\n{'has_search_bar': True, 'clickables': ['search']}",
                "Observation:\nWebShop [SEP] Instruction: [SEP] i'm looking for a pair of water resistant brown pants, and price lower than 60.00 dollars [SEP] Search\n\nAvailable Actions:\n{'has_search_bar': True, 'clickables': ['search']}"
            ]
        else:
            raise NotImplementedError()


        self.raw_dataset = DatasetDict({
            'validation': Dataset.from_dict({
                'context': [self.agent_prompt],
                'context_id': [0],
                'input': [''],
                'output': ['']
            })
        })
        
        if dataset_mode=='pseudo':
            self.ds = self._load_pseudo_dataset(load_dataset_path)
        elif dataset_mode=='original':
            self.ds = self.raw_dataset['validation']
        # elif dataset_mode=='original_sample':
        #     self.ds = self.raw_dataset['validation'].select([0, 222])
        else:
            raise NotImplementedError()

        self.pseudo_input_demons = None

    def _make_pseudo_input_demons(self, numbered_list_format):
        if len(self.pseudo_input_demons_list)==0:
            return ''
        return self.pseudo_input_user_prompt.format(
            num_gen_once=len(self.pseudo_input_demons_list),
            inputs='\n'.join([f"{numbered_list_format.format(num=i+1)}{_input}" for i, _input in enumerate(self.pseudo_input_demons_list)])
        )
    
    ### pseudo input
    def get_pseudo_input_system_prompt(self, numbered_list_format='{num}.'):
        if self.pseudo_input_demons==None:
            self.pseudo_input_demons = self._make_pseudo_input_demons(numbered_list_format)
        return self.pseudo_input_system_prompt.format(context=self.agent_prompt) + self.pseudo_input_demons
        
    def get_pseudo_input_user_prompt(self, context, num_gen_inputs):
        return self.pseudo_input_user_prompt.format(num_gen_once=num_gen_inputs, inputs='')
    
    ### student & teacher
    def get_student_input_prompt(self, input):
        return self.pseudo_student_input_prompt.format(input=input)
    
    def get_teacher_input_prompt(self, context, input):
        return self.pseudo_teacher_input_prompt.format(context=context, input=input)
    
    ### reason
    def get_reason_system_prompt(self):
        return self.reason_system_prompt
    
    def get_reason_user_prompt(self, context, input, student_output, teacher_output):
        return self.reason_user_prompt.format(context=context, input=input, student_output=student_output, teacher_output=teacher_output)
    
    ### meta-cognition
    def get_meta_cognition_input_prompt(self, input, student_output, teacher_output):
        return self.meta_cognition_input_prompt.format(input=input, student_output=student_output, teacher_output=teacher_output)
    
    def get_meta_cognition_output_prompt(self, context, reason):
        return self.meta_cognition_output_prompt.format(context=context, reason=reason)

    ### save & load
    def save_pseudo_dataset(self, path, context_id_list, context_list, pseudo_input_list, teacher_output_list=None, student_output_list=None, reason=None, conv_list=None):
        assert len(context_list)==len(pseudo_input_list), f"{len(context_list)}, {len(pseudo_input_list)}"
        _dic = {
            'context_id': context_id_list,
            'context':context_list,
            'pseudo_input': pseudo_input_list
        }

        if teacher_output_list!=None and student_output_list!=None:
            _dic['teacher_output_single'] = teacher_output_list
            _dic['student_output_single'] = student_output_list
        if reason!=None:
            _dic['reason'] = reason
        if conv_list!=None:
            _dic['teacher_output_conv'] = conv_list
        
        ds = Dataset.from_dict(_dic)
        ds.save_to_disk(Path(get_project_root(), path).as_posix())

    def _load_pseudo_dataset(self, path):
        return datasets.load_from_disk(Path(get_project_root(), path).as_posix())
    
    def evaluate(self, pred_list, current_ds):
        # evaluator = EvaluateTool()
        # metrics = evaluator.evaluate(pred_list, current_ds)
        # return dict(metrics)
        pass
