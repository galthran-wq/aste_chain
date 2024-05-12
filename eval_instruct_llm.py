# %%
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import fire

def main(checkpoint):
    base_model_id = "/external/nfs/lamorozov/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/b70aa86578567ba3301b21c8a27bea4e8f6d6d61"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=False,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    import datasets
    dataset = datasets.load_from_disk("./data/hg/banks_sentenized_w_emp")

    # %%
    from gigachain_extensions.pydantic_models import ASTEAnswer
    ASTEAnswer(triplets=dataset['train']['triplets'][11]).model_dump_json()

    # %%
    from peft import PeftModel

    ft_model = PeftModel.from_pretrained(model, 
                                         checkpoint
                                        #  "/external/nfs/lamorozov/aste_chain/mistral-instruct-finetune-json/checkpoint-2819"
                                         )

    # %%
    ft_model.eval()

    # %%
    def formatting_func(text):
        """Gen. input text based on a prompt, task instruction, (context info.), and answer
    
        :param data_point: dict: Data point
        :return: dict: tokenzed prompt
        """
        prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                'appropriately completes the request.\n\n'
        instruction = """Ты занимаешься аспектно-ориентированным анализом настроения клиентов.
    Ты выделяешь из отзывов клиентов полярности для терминов аспектов (aspect term) и терминов мнения (opinion term).
    Термины аспектов (aspect terms) -- конкретного элемента или характеристика товара, продукта или сервиса, которую анализируют для определения настроения или отношения. Аспектами могут быть: качество, цена, удобство использования и т.д..
    Термины мнения (opinion terms) -- выражения, отражающие отношение клиента к аспекту.
    Полярность (sentiment polarity) -- отношение клиента к аспекту, выражаемое терминами мнения. Примает одно значение из "POS" или "NEG".

    Твой ответ обязательно должен соответствовать формату JSON. Схема ответа:
    {{
        "triplets": [
            {{
                // Характеристика
                "aspect_term": string,
                // Термины мнения
                "opinion_term": string,
                // Полярность
                "sentiment": "POS" или "NEG",
            }},
            ...
        ]
    }}
    """
        text = f"""<s>[INST]{prefix_text} {instruction} Вход: {text} [/INST]Ответ:"""
        return text

    # %%
    import pickle
    # import pickle
    # with open("answer.pickle", "rb") as f:
    #     answers = pickle.load(f)

    # %%
    answers = []

    # %%
    from tqdm.auto import tqdm

    def batch(iterable, n=16):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    with torch.no_grad():
        for example in tqdm(batch(dataset['val']['text'][len(answers):], n=8), total=len(dataset['val'][len(answers):]) // 8):
            eval_prompts = [formatting_func(example_) for example_ in example]
            model_input = tokenizer(eval_prompts, return_tensors="pt", truncation=True, max_length=600, padding=True, ).to("cuda")
            gen = ft_model.generate(**model_input, max_new_tokens=400, num_beams=5)
            answers.extend(tokenizer.decode(answer, skip_special_tokens=True) for answer in gen)

    # %%
    import pickle
    with open(checkpoint + "/answer_val.pickle", "wb") as f:
        pickle.dump(answers, f)

    # %%
    print(answers[0])

    # %%
    import json
    from gigachain_extensions.pydantic_models import ASTEAnswer
    parsed_answers = []
    def parse_answer(answer: str):
        start_answer = answer[answer.find("```json")+7:]
        answer = start_answer[:start_answer.find("```")]
        try:
            return ASTEAnswer.model_validate_json(answer)
        except:
            return ASTEAnswer(triplets=[])

    def parse_answer_ft(answer: str):
        answer = answer[answer.find("Ответ:")+6:]
        try:
            return ASTEAnswer.model_validate_json(answer)
        except:
            return ASTEAnswer(triplets=[])


    # answers2 = [answer[answer.find("[/INST]"):].strip() for answer in answers]
    # answers2 = [answer[answer.find("```"):].strip()[:-3] for answer in answers2]
    for answer in answers:
        parsed_answers.append((parse_answer_ft(answer)))

    # %%
    # import pickle
    # with open("answer.pickle", "rb") as f:
    #     zeroshot_answers = pickle.load(f)

    # %%
    # zeroshot_parsed_answers = [parse_answer(answer) for answer in zeroshot_answers]

    # %%
    print(eval(parsed_answers, dataset["val"]))


if __name__ == "__main__":
    fire.Fire(main)