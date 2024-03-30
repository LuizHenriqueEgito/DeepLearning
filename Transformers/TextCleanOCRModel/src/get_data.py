from pathlib import Path
from typing import Dict

def get_data() -> Dict[str, str]:
    input_data = Path('data/inputs/text1_dirty.txt')
    tartget_data = Path('data/targets/text1_clean.txt')
    dirty_text = input_data.read_text(encoding='utf8')
    clean_text = tartget_data.read_text(encoding='utf8')
    return {'input': dirty_text, 'target': clean_text}
