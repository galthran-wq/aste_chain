from langchain.prompts import ChatPromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate

AOP_FORMAT_TEMPLATE = """Твой ответ обязательно должен соответствовать формату JSON. Схема ответа:
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

AO_FORMAT_TEMPLATE = """Твой ответ обязательно должен соответствовать формату JSON. Схема ответа:
{{
    "duplets": [
        {{
            // Характеристика
            "aspect_term": string,
            // Термины мнения
            "opinion_term": string,
        }},
        ...
    ]
}}
"""

A_FORMAT_TEMPLATE = """Твой ответ обязательно должен соответствовать формату JSON. Схема ответа:
{{
    "aspects": [
        {{
            // Характеристика
            "aspect_term": string,
        }},
        ...
    ]
}}
"""


def get_fewshot_gen_aspect_prompt(examples):
    system_prompt = f"""
Ты -- опытный работник банка. Твоя задача понимать, что людям нравится или не нравится в работе банка. Для этого ты занимаешься аспектно-ориентированным анализом настроения клиентов.
Ты выделяешь из отзывов клиентов термины аспектов (aspect term).
Термины аспектов (aspect terms) -- характеристики конкретного элемента или товара, продукта или сервиса, которую анализируют для определения настроения или отношения. Аспектами могут быть: качество, цена, удобство использования и т.д..

Условия:
- Термины аспектов должны содержаться в тексте отзыва.

{A_FORMAT_TEMPLATE}
"""
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "Отзыв:\n{text}"),
        ("ai", "{aspects}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("user", "Отзыв:\n{text}")
        ]
    )
    return final_prompt


def get_fewshot_gen_aspect_opinion_prompt(examples=None, example_selector=None):
    system_prompt = f"""
Ты -- опытный работник банка. Твоя задача понимать, что людям нравится или не нравится в работе банка. Для этого ты занимаешься аспектно-ориентированным анализом настроения клиентов.
Ты выделяешь из отзывов клиентов термины аспектов (aspect term) и термины мнения (opinion term).
Термины аспектов (aspect terms) -- характеристики конкретного элемента или характеристика товара, продукта или сервиса, которую анализируют для определения настроения или отношения. Аспектами могут быть: качество, цена, удобство использования и т.д..
Термины мнения (opinion terms) -- выражения, отражающие отношение клиента к аспекту.

Условия:
- Термины аспектов и термины мнений должны содержаться в тексте отзыва.

{AO_FORMAT_TEMPLATE}
"""
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "Отзыв:\n{text}"),
        ("ai", "{duplets}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        example_selector=example_selector,
        input_variables=["text", "duplets"]
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("user", "Отзыв:\n{text}")
        ]
    )
    return final_prompt


def get_fewshot_gen_opinion_from_aspect_prompt(examples=None, example_selector=None):
    system_prompt = f"""
Ты -- опытный работник банка. Твоя задача понимать, что людям нравится или не нравится в работе банка. Для этого ты занимаешься аспектно-ориентированным анализом настроения клиентов.
Ты выделяешь из отзывов клиентов термины мнения (aspect term) для терминов аспекта (opinion terms).
Термины аспектов (aspect terms) --  характеристики конкретного элемента или характеристика товара, продукта или сервиса, которую анализируют для определения настроения или отношения. Аспектами могут быть: качество, цена, удобство использования и т.д..
Термины мнения (opinion terms) -- выражения, отражающие отношение клиента к аспекту.

Условия:
- Термины мнений должны содержаться в тексте отзыва.
- В твоем ответе должны содержаться все аспекты, перечисленные в запросе пользователя. Аспекты должны точно совпадать с указанными в примере.

{AO_FORMAT_TEMPLATE}
"""
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "Отзыв:\n{text}\nСписок терминов аспектов:\n{aspects}"),
        ("ai", "{duplets}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        example_selector=example_selector,
        input_variables=["text", "aspects", "duplets"]
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("user", "Отзыв:\n{text}\nСписок терминов аспектов:\n{aspects}")
        ]
    )
    return final_prompt


def get_fewshot_gen_polarity_from_aspects_opinions_prompt(examples):
    system_prompt = f"""
Ты -- опытный работник банка. Твоя задача понимать, что людям нравится или не нравится в работе банка. Для этого ты занимаешься аспектно-ориентированным анализом настроения клиентов.
Ты выделяешь из отзывов клиентов полярности для терминов аспектов (aspect term) и терминов мнения (opinion term).
Термины аспектов (aspect terms) -- конкретного элемента или характеристика товара, продукта или сервиса, которую анализируют для определения настроения или отношения. Аспектами могут быть: качество, цена, удобство использования и т.д..
Термины мнения (opinion terms) -- выражения, отражающие отношение клиента к аспекту.
Полярность (sentiment polarity) -- отношение клиента к аспекту, выражаемое терминами мнения. Примает одно значение из "POS" или "NEG".

Условия:
- В твоем ответе должны содержаться все аспекты, перечисленные в запросе пользователя. Аспекты и термины мнения должны точно совпадать с указанными в примере.

{AOP_FORMAT_TEMPLATE}
"""
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "Отзыв:\n{text}\nСписок терминов аспектов и терминов мнения:\n{duplets}"),
        ("ai", "{triplets}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("user", "Отзыв:\n{text}\nСписок терминов аспектов и терминов мнения:\n{duplets}")
        ]
    )
    return final_prompt


def get_fewshot_aop_prompt(examples=None, example_selector=None):
    system_prompt = f"""
Ты -- опытный работник банка. Твоя задача понимать, что людям нравится или не нравится в работе банка. Для этого ты занимаешься аспектно-ориентированным анализом настроения клиентов.
Ты выделяешь из отзывов клиентов термины аспектов (aspect term), термины мнения (opinion term), и полярности настроения.
Термины аспектов (aspect terms) -- конкретного элемента или характеристика товара, продукта или сервиса, которую анализируют для определения настроения или отношения. Аспектами могут быть: качество, цена, удобство использования и т.д..
Термины мнения (opinion terms) -- выражения, отражающие отношение клиента к аспекту.
Полярность (sentiment polarity) -- примает одно значение из "POS" или "NEG".

Условия:
- Термины аспектов и термины мнений должны содержаться в тексте отзыва.

{AOP_FORMAT_TEMPLATE}
"""
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "Отзыв:\n{text}\nСписок терминов аспектов, терминов полярности и полярностей из отзыва:"),
        ("ai", "{triplets}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        example_selector=example_selector,
        input_variables=["text", "triplets"]
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("user", "Отзыв:\n{text}\nСписок терминов аспектов, терминов полярности и полярностей из отзыва:")
        ]
    )
    return final_prompt


# def get_fewshot_gen_polarity_prompt(examples):
#     system_prompt = """
# Ты -- опытный работник банка. Твоя задача понимать, что людям нравится или не нравится в работе банка. Для этого ты занимаешься аспектно-ориентированным анализом настроения клиентов.
# Ты выделяешь из отзывов клиентов полярности для терминов аспектов (aspect term) и терминов мнения (opinion term).
# Термины аспектов (aspect terms) -- конкретного элемента или характеристика товара, продукта или сервиса, которую анализируют для определения настроения или отношения. Аспектами могут быть: качество, цена, удобство использования и т.д..
# Термины мнения (opinion terms) -- выражения, отражающие отношение клиента к аспекту.
# Полярность (sentiment polarity) -- . Примает одно значение из "POS" или "NEG".

# Условия:
# - Термины аспектов и термины мнений должны содержаться в тексте отзыва.
# """
#     example_prompt = ChatPromptTemplate.from_messages([
#         ("user", "Отзыв:\n{text}\nСписок терминов аспектов и терминов полярности:\n{ao_duplets}"),
#         ("ai", "{triplets}"),
#     ])
#     few_shot_prompt = FewShotChatMessagePromptTemplate(
#         example_prompt=example_prompt,
#         examples=examples,
#     )
#     final_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             few_shot_prompt,
#             ("user", "Отзыв:\n{text}\nСписок терминов аспектов и терминов полярности:\n{ao_duplets}")
#         ]
#     )
#     return final_prompt
