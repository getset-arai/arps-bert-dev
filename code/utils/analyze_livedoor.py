import os
import re


def remove_brackets(inp):
    brackets_tail = re.compile('【[^】]*】$')
    brackets_head = re.compile('^【[^】]*】')
    output = re.sub(brackets_head, '', re.sub(brackets_tail, '', inp))
    return output


class AnalyzeLivedoor:
    def __init__(self, dir_name: str):
        self.dir_name = dir_name

    def get_data_list(self, ) -> list[str]:
        data_list = []
        for file_name in os.listdir(self.dir_name):
            if file_name == "LICENSE.txt":
                continue
            with open(os.path.join(self.dir_name, file_name), encoding="utf-8") as f:
                lines = f.read().splitlines()
                text = ""
                for th_line in lines[3:]:  # 4行目以降が本文
                    if "■関連記事" in th_line or "■関連リンク" in th_line:
                        break
                    else:
                        text += f"{th_line}\n"
                data_list.append(text)
        return data_list
