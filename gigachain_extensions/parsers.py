from ast import literal_eval

from langchain.output_parsers.list import ListOutputParser


class AnyListOutputParser(ListOutputParser):
    def parse(self, text):
        text = text.replace("[", "(")
        text = text.replace("]", ")")
        try:
            return literal_eval(text)
        except Exception as e:
            return None
