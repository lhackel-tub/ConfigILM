Search.setIndex({"docnames": ["extra/BEN_LMDB_Reader", "extra/bigearthnet", "extra/cocoqa", "extra/rsvqaxben", "general/code_of_conduct", "general/contributing", "general/dependencies", "general/license", "ilmconfiguration", "image_captioning", "index", "installation", "references", "sup_pretraining", "vqa", "wip"], "filenames": ["extra/BEN_LMDB_Reader.ipynb", "extra/bigearthnet.ipynb", "extra/cocoqa.ipynb", "extra/rsvqaxben.ipynb", "general/code_of_conduct.md", "general/contributing.md", "general/dependencies.md", "general/license.md", "ilmconfiguration.ipynb", "image_captioning.ipynb", "index.md", "installation.ipynb", "references.md", "sup_pretraining.ipynb", "vqa.ipynb", "wip.md"], "titles": ["Using the BigEarthNet LMDB Reader", "BigEarthNet Dataset &amp; Datamodule", "COCO-QA", "RSVQAxBEN Dataset &amp; Datamodule", "Contributor Covenant Code of Conduct", "Contributing", "Dependencies", "License", "Model Configuration", "[WIP] Image Captioning", "ConfigILM", "Installation", "Further references", "Supervised Image Classification", "Visual Question Answering (VQA)", "&lt;no title&gt;"], "terms": {"In": [0, 1, 2, 3, 5, 12], "thi": [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14], "section": [0, 11, 13, 14], "an": [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15], "exampl": [0, 3, 4, 8, 10, 11, 13, 14], "i": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15], "shown": [0, 14], "convert": [0, 3], "lightn": [0, 1, 3, 6, 10], "memori": [0, 1, 3], "map": 0, "databas": 0, "manag": [0, 5, 13, 14], "background": 0, "index": [0, 1, 2, 3], "python": [0, 5, 10, 13, 14], "object": [0, 1, 3, 8], "due": 0, "its": [0, 1, 2, 3, 5], "which": [0, 1, 3, 4, 8, 10, 11, 14], "pickl": 0, "abl": [0, 3], "thread": 0, "safe": 0, "after": [0, 1, 3, 5, 13, 14], "first": [0, 1, 2, 3, 12, 13, 14], "access": [0, 1, 3, 12], "howev": [0, 1, 2, 3], "onli": [0, 1, 2, 3, 13, 14], "fork": 0, "support": [0, 1, 3, 8, 10], "e": [0, 4, 11, 13, 14], "g": [0, 11, 13, 14], "__getitem__": 0, "method": 0, "pytorch": [0, 1, 3, 6, 10], "dataset": [0, 11, 12], "To": [0, 1, 3, 5, 13, 14], "we": [0, 2, 3, 4, 5, 12, 13, 14], "have": [0, 1, 3, 4, 5, 13, 14], "creat": [0, 5, 8, 10], "benlmdbread": [0, 1, 3], "need": [0, 1, 2, 3, 5, 13, 14], "4": [0, 1, 3, 6, 10, 12, 13, 14], "argument": 0, "creation": [0, 10, 13, 14], "name": [0, 1, 3, 8, 12], "directori": [0, 5], "where": [0, 10, 14], "lmbd": 0, "file": [0, 1, 3, 7], "locat": 0, "string": 0, "sequenc": [0, 3, 8], "3": [0, 1, 2, 3, 6, 8, 12, 13, 14], "int": [0, 8, 14], "desir": 0, "imag": [0, 2, 8, 10, 11, 12, 14], "size": [0, 1, 2, 3, 4, 8, 14], "channel": [0, 1, 2, 3, 4, 8, 13, 14], "height": 0, "width": [0, 2], "indic": 0, "ar": [0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 13, 14], "from": [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14], "configilm": [0, 1, 2, 3, 5, 8, 9, 11, 13, 14, 15], "extra": [0, 1, 2, 3, 6, 11, 13, 14], "ben_lmdb_util": 0, "import": [0, 1, 2, 3, 8, 13, 14], "ben_read": 0, "lmdb_dir": 0, "my_data_path": [0, 1, 2, 3, 13, 14], "path": [0, 1, 2, 3, 13, 14], "image_s": [0, 1, 3, 8, 13, 14], "120": [0, 1, 2, 3, 8, 10, 13, 14], "rgb": [0, 1, 3, 13, 14], "label_typ": 0, "old": 0, "img": [0, 1, 2, 3], "lbl": [0, 1], "s2b_msil2a_20180502t093039_82_40": 0, "expect": [0, 1, 2, 3, 13, 14], "contain": [0, 1, 3, 14], "3x120x120": 0, "annot": 0, "43": [0, 12], "version": [0, 1, 4, 6, 10], "deliv": 0, "torch": [0, 1, 2, 3, 6, 8, 13, 14], "tensor": [0, 1, 2, 3, 8, 14], "list": [0, 5, 13, 14], "complex": 0, "cultiv": 0, "pattern": [0, 4], "land": [0, 1], "princip": 0, "occupi": 0, "agricultur": [0, 3, 14], "signific": 0, "area": [0, 3, 5, 10, 14], "natur": [0, 2, 3, 4], "veget": 0, "broad": 0, "leav": 0, "forest": [0, 3, 14], "transit": [0, 3], "woodland": [0, 3], "shrub": [0, 3], "water": 0, "bodi": [0, 4], "If": [0, 2, 3, 5, 10, 11], "now": [0, 9, 13, 14, 15], "interest": 0, "can": [0, 1, 2, 3, 5, 8, 10, 11, 14], "specif": [0, 1], "return": [0, 1, 13, 14], "b8": 0, "b4": 0, "The": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15], "defin": [0, 3, 13, 14], "begin": 0, "align": [0, 4], "vi": 0, "frac": [0, 1, 3], "b08": 0, "b04": 0, "end": [0, 1, 3, 5, 11, 14], "2": [0, 1, 2, 3, 6, 8, 12, 14], "veg_idx": 0, "0": [0, 1, 2, 3, 4, 6, 8, 11, 13, 14], "1": [0, 1, 2, 3, 6, 10, 11, 12, 13, 14], "08": 0, "dimens": [0, 3], "04": 0, "like": [0, 1, 3, 4, 5, 8], "order": [0, 1, 3], "specifi": [0, 1, 3, 4, 8], "paramet": [0, 1, 2, 3, 8, 13, 14], "note": [0, 1, 3], "also": [0, 1, 3, 4, 8], "ha": [0, 1, 3, 11], "set": [0, 2, 4, 5, 11, 12, 13, 14], "check": [0, 5, 11], "interpol": 0, "alreadi": [0, 14], "appli": [0, 4], "loader": 0, "nn": [0, 13, 14], "function": [0, 1, 3, 11, 13, 14], "corner": 0, "bicub": 0, "mode": 0, "For": [0, 1, 3, 4, 8, 11, 13, 14], "eas": [0, 10], "some": [0, 3, 5, 11, 13, 14], "predefin": [0, 11], "configur": [0, 1, 2, 3, 11], "avail": [0, 1, 4], "without": [0, 3, 4, 7], "all": [0, 1, 3, 4, 5, 7, 8, 11, 12, 13, 14], "pre": [0, 6, 10, 11, 13, 14], "definit": 0, "respect": [0, 1, 4, 8, 13, 14], "s1": [0, 12], "vh": 0, "vv": 0, "s2": [0, 1, 3, 12], "b02": 0, "b03": 0, "b05": 0, "b06": 0, "b07": 0, "b11": 0, "b12": 0, "b8a": 0, "10m20m": 0, "ir": 0, "10": [0, 1, 3, 6, 8, 10, 12, 14], "12": [0, 1, 3, 6, 13, 14], "request": [0, 4, 5, 10], "new": [0, 12], "19": [0, 1, 3, 12, 13], "introduc": [0, 12], "sumbul": [0, 1, 12], "et": [0, 1, 2, 3, 12, 14], "al": [0, 1, 2, 3, 12, 14], "here": [0, 1, 3, 12, 13, 14], "see": [0, 1, 2, 3, 4], "get": [0, 5, 11, 14], "inland": 0, "pprint": [0, 8], "wish": 0, "dimension": 0, "one": [0, 1, 2, 3, 14], "hot": [0, 1, 3], "guarante": 0, "uniform": 0, "convers": 0, "so": [0, 7, 14], "each": [0, 1, 3], "vector": [0, 1, 3], "alwai": [0, 3, 8, 14], "same": [0, 1, 3, 14], "regardless": [0, 4], "user": [0, 10, 14], "ben19_list_to_onehot": 0, "collect": [0, 8, 10, 12], "dure": [0, 3, 4, 13, 14], "initi": [0, 1, 14], "base": [0, 1, 2, 3], "chosen": 0, "ben_reader_1": 0, "print": [0, 1, 3], "f": [0, 3, 13, 14], "std": 0, "ben_reader_2": 0, "b01": 0, "2218": 0, "94553375": 0, "590": 0, "23569706": 0, "1365": 0, "45589904": 0, "675": 0, "88746967": 0, "340": 0, "76769064": 0, "2266": 0, "46036911": 0, "554": 0, "81258967": 0, "1356": 0, "13789355": 0, "page": [1, 2, 3, 5, 10, 11], "describ": [1, 2, 3, 13, 14], "usag": [1, 2, 3, 13, 14], "multi": [1, 12], "spectral": 1, "multilabel": [1, 12], "remot": [1, 10, 12], "sens": [1, 10, 12], "us": [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14], "cover": [1, 3, 11], "classif": [1, 8, 12], "offici": [1, 4], "paper": [1, 12], "wa": [1, 2, 3, 4, 14], "publish": [1, 2, 3, 4, 7, 10, 14], "updat": 1, "modal": [1, 12, 14], "detail": [1, 10], "inform": [1, 4, 8, 12], "itself": 1, "pleas": [1, 10], "refer": 1, "public": [1, 4, 12], "guid": [1, 5], "modul": [1, 3], "two": [1, 3, 10, 11], "class": [1, 4, 8, 12, 13, 14], "standard": [1, 3], "util": [1, 3], "data": [1, 2, 3, 12, 13, 14], "pytorch_lightn": [1, 3, 13, 14], "lightningdatamodul": [1, 3], "encapsul": [1, 3, 13, 14], "easi": [1, 3, 10, 14], "applic": [1, 3, 10], "read": [1, 3, 12], "label": [1, 13, 14], "lmdb": [1, 3, 6], "most": [1, 2, 3, 10, 14], "basic": [1, 2, 3, 13], "form": [1, 2, 3, 14], "csv": [1, 3], "assum": [1, 3, 11], "bigearthnetencod": [1, 3], "o": [1, 3], "point": [1, 3], "view": [1, 3], "folder": [1, 2, 3], "10m": [1, 3], "20m": [1, 3], "sentinel": [1, 3, 12], "full": [1, 2, 3, 14], "structur": [1, 2, 3], "mdb": [1, 3], "lock": [1, 3], "test": [1, 2, 3, 5, 10, 13, 14], "train": [1, 2, 3, 11, 12, 13, 14], "val": [1, 3, 13, 14], "ben_datamodule_lmdb_encod": [1, 13], "d": [1, 2, 3], "root_dir": [1, 2, 3], "56": 1, "flip": [1, 3], "choos": [1, 3], "bgr": [1, 3], "axi": [1, 3], "displai": [1, 3, 13, 14], "bring": [1, 3], "home": [1, 14], "runner": [1, 14], "work": [1, 5, 10, 13, 14], "venv": 1, "lib": 1, "python3": 1, "site": [1, 10], "packag": [1, 3, 10, 13, 14], "tqdm": 1, "auto": [1, 13, 14], "py": [1, 14], "21": 1, "tqdmwarn": 1, "iprogress": 1, "found": [1, 3], "jupyt": 1, "ipywidget": 1, "http": [1, 4, 5, 10, 12], "readthedoc": 1, "io": 1, "en": 1, "stabl": 1, "user_instal": 1, "html": [1, 4], "autonotebook": 1, "notebook_tqdm": 1, "ben": 1, "none": [1, 2, 3, 8, 11, 13, 14], "75": [1, 3], "patch": [1, 12], "filter": 1, "differ": [1, 3, 4, 13, 14], "via": [1, 3, 4, 10, 11], "limit": [1, 2, 3, 7], "combin": [1, 3, 8, 10, 11, 14], "call": [1, 3, 13, 14], "get_available_channel_configur": 1, "altern": [1, 11], "faulti": [1, 3], "possibl": [1, 3, 5, 8, 10], "well": [1, 4, 8, 13, 14], "whilst": 1, "rais": 1, "assertionerror": [1, 3], "lmdbreader": 1, "It": [1, 2, 3, 9, 15], "By": [1, 3], "default": [1, 2, 3, 8], "three": [1, 3], "_": [1, 2, 3], "25": [1, 2, 3, 8], "max_img_idx": [1, 2, 3], "n": [1, 3], "alphabet": [1, 3], "A": [1, 3, 4, 7, 8, 12], "larger": [1, 3], "than": [1, 3], "": [1, 3, 4, 10, 11], "case": [1, 3, 11], "equal": [1, 3], "behaviour": [1, 3], "100": [1, 2, 3, 10], "wrap": [1, 3], "automat": [1, 3, 5, 13, 14], "gener": [1, 2, 3, 5, 13, 14], "per": [1, 3], "augment": [1, 3], "shuffl": [1, 3], "etc": [1, 3], "depend": [1, 3, 5, 8, 11, 13, 14], "resiz": [1, 3], "normal": [1, 3, 13, 14], "addition": [1, 3, 10, 14], "nois": [1, 3], "rotat": [1, 3], "overwritten": [1, 3], "below": [1, 3, 8], "setup": [1, 3], "popul": [1, 3], "insid": [1, 3, 13, 14], "stage": [1, 3], "fit": [1, 3, 7, 13, 14], "prepar": [1, 3], "valid": [1, 3, 13, 14], "dm": [1, 3, 13, 14], "data_dir": [1, 3, 13, 14], "befor": [1, 3], "train_d": [1, 3], "val_d": [1, 3], "test_d": [1, 3], "worker": [1, 3, 13, 14], "15": 1, "took": 1, "00": [1, 3], "second": 1, "total": [1, 3], "sampl": 1, "0x7f1ed25423e0": 1, "0x7f1fa2d46860": 1, "afterward": [1, 3], "easili": [1, 3, 11], "len": [1, 3], "dl": [1, 3], "lceil": [1, 3], "batch": [1, 3, 13, 14], "_size": [1, 3], "rceil": [1, 3], "therefor": [1, 3, 8, 14], "batch_siz": [1, 3], "16": [1, 3], "train_load": [1, 3], "train_dataload": [1, 3], "addit": [1, 3, 10, 11, 14], "pass": [1, 3, 5, 13, 14], "through": [1, 3, 4, 10], "settabl": [1, 3], "num_workers_dataload": [1, 3, 13, 14], "cpu_count": [1, 3], "valu": [1, 3, 8, 13, 14], "mean": [1, 3, 11, 13, 14], "chang": [1, 3, 5], "accompani": [1, 3], "messag": [1, 3], "hint": [1, 3], "Not": [1, 8], "changeabl": 1, "pin": [1, 3], "true": [1, 3, 6, 8, 13, 14], "cuda": [1, 3], "enabl": [1, 10], "devic": [1, 3, 13, 14], "fals": [1, 3, 8, 13, 14], "otherwis": [1, 3, 4, 7], "7": [1, 3, 6], "96mhint": [1, 3, 14], "recommend": [1, 3, 11, 12], "0m": [1, 2, 3, 14], "dataload": [2, 10, 13, 14], "datamodul": [2, 10, 13, 14], "vqa": [2, 3, 11], "ren": [2, 12], "coco_train2014_": 2, "id": [2, 3], "jpg": 2, "b": 2, "coco_val2014_": 2, "j": 2, "z": 2, "qa_qa_test": 2, "json": [2, 3], "qa_qa_train": 2, "cocoqa_datamodul": 2, "cocoqadataset": 2, "question": [2, 3, 4, 12], "answer": [2, 4, 12], "93mwarn": 2, "No": [2, 4], "token": [2, 14], "provid": [2, 3, 4, 5, 7, 10, 12], "berttoken": 2, "uncas": 2, "mai": [2, 3, 4, 5, 14], "result": [2, 3, 13, 14], "veri": 2, "bad": [2, 3], "perform": [2, 3], "network": [2, 8, 13, 14], "other": [2, 3, 4, 5, 7, 8, 12, 13, 14], "load": [2, 14], "50": 2, "pair": [2, 3, 12], "reduc": [2, 3], "start": [2, 3, 5, 11, 13, 14], "101": [2, 3, 14], "2054": 2, "2003": 2, "1996": 2, "3609": 2, "1997": 2, "3933": 2, "102": [2, 3, 14], "As": [2, 3], "out": [2, 3, 7, 11], "languag": [2, 3, 4, 8, 10, 11, 14], "text": [2, 3, 14], "lead": [2, 3, 4], "account": [2, 3, 4], "input": [2, 3, 13, 14], "get_hf_model": [2, 3, 14], "prajjwal1": [2, 3, 14], "bert": [2, 3, 14], "tini": [2, 3, 14], "split": 2, "transform": [2, 6], "number": [2, 13, 14], "img_siz": [2, 3, 13, 14], "should": [2, 8, 10], "high": 2, "seq_length": [2, 3], "200": 2, "5": [2, 3, 6], "32": [2, 8], "3157": 2, "2417": 2, "1012": 2, "bigearthnet": [3, 6, 13], "lobri": [3, 12, 14], "zenodo": [3, 10], "context": 3, "extend": 3, "includ": [3, 4, 7, 10, 12], "origin": 3, "small": 3, "distribut": [3, 7], "vqa_rsvqaxben": 3, "rsvqaxben_qa_train": 3, "rsvqaxben_qa_v": 3, "rsvqaxben_qa_test": 3, "rsvqaxben_datamodule_lmdb_encod": [3, 14], "2024": [3, 14], "2070": 3, "6138": [3, 14], "2556": [3, 14], "1029": [3, 14], "integ": 3, "length": 3, "shorter": 3, "pad": [3, 14], "zero": 3, "longer": 3, "truncat": 3, "last": 3, "non": [3, 13, 14], "question1": 3, "7976": 3, "2752": [3, 14], "2030": 3, "4910": [3, 14], "8": 3, "question2": 3, "reconstruct": 3, "special": 3, "decod": [3, 14], "cl": [3, 14], "artifici": 3, "present": [3, 11, 14], "sep": [3, 14], "skip_special_token": 3, "current": [3, 8, 11], "preconfigur": 3, "when": [3, 4, 14], "try": [3, 5, 11, 14], "except": 3, "qa": 3, "np": 3, "arrai": [3, 10], "69672": 3, "82it": 3, "311150": 3, "15it": 3, "relev": [3, 12], "certain": 3, "output": [3, 13, 14], "could": [3, 4], "up": [3, 5, 11, 13, 14], "million": 3, "prevent": [3, 8], "explos": 3, "determin": [3, 4], "highest": 3, "reduct": 3, "count": 3, "172747": 3, "28it": 3, "frequent": 3, "about": [3, 4, 5], "96": 3, "149796": 3, "57it": 3, "These": [3, 12, 13, 14], "re": [3, 12], "subset": 3, "requir": [3, 5, 8, 10, 12], "doe": 3, "necessarili": 3, "match": 3, "fewer": 3, "never": 3, "encod": [3, 6], "more": [3, 5, 12], "indexerror": 3, "happen": 3, "element": [3, 14], "selected_answ": 3, "ye": [3, 14], "mix": 3, "0x7fb16ff37520": 3, "0x7fb16ff36b30": 3, "four": 3, "pin_memori": [3, 14], "overwrit": 3, "member": 4, "leader": 4, "make": [4, 10, 12, 14], "particip": 4, "commun": [4, 5, 10], "harass": 4, "free": [4, 5, 7, 10], "experi": [4, 10], "everyon": [4, 5], "ag": 4, "visibl": 4, "invis": 4, "disabl": 4, "ethnic": 4, "sex": 4, "characterist": 4, "gender": 4, "ident": [4, 14], "express": [4, 7], "level": [4, 12], "educ": 4, "socio": 4, "econom": [4, 10], "statu": [4, 10], "nation": 4, "person": [4, 7], "appear": 4, "race": 4, "religion": 4, "sexual": 4, "orient": 4, "act": 4, "interact": [4, 12], "wai": [4, 5, 11, 14], "contribut": 4, "open": [4, 5, 10], "welcom": [4, 5], "divers": [4, 10], "inclus": 4, "healthi": 4, "behavior": 4, "posit": 4, "environ": [4, 5], "demonstr": [4, 10], "empathi": 4, "kind": [4, 7], "toward": 4, "peopl": 4, "Being": 4, "opinion": 4, "viewpoint": 4, "give": 4, "gracefulli": 4, "accept": 4, "construct": 4, "feedback": 4, "apolog": 4, "those": 4, "affect": 4, "mistak": [4, 5], "learn": [4, 6], "focus": 4, "what": 4, "best": [4, 5], "just": [4, 13, 14], "u": [4, 13, 14], "individu": [4, 10], "overal": 4, "unaccept": 4, "imageri": 4, "attent": 4, "advanc": [4, 12], "ani": [4, 7, 10, 11], "troll": 4, "insult": 4, "derogatori": 4, "comment": 4, "polit": 4, "attack": 4, "privat": 4, "physic": 4, "email": 4, "address": 4, "explicit": 4, "permiss": [4, 7], "reason": 4, "consid": [4, 10], "inappropri": 4, "profession": 4, "clarifi": 4, "take": [4, 5], "appropri": 4, "fair": 4, "action": [4, 7, 10], "thei": [4, 10], "deem": 4, "threaten": 4, "offens": 4, "harm": 4, "right": [4, 7], "remov": [4, 13, 14], "edit": 4, "reject": 4, "commit": [4, 6], "wiki": 4, "issu": [4, 5, 10], "moder": 4, "decis": 4, "within": [4, 10], "space": [4, 10], "repres": 4, "mail": 4, "post": 4, "social": 4, "media": 4, "appoint": 4, "onlin": 4, "offlin": 4, "event": [4, 7], "instanc": 4, "abus": 4, "report": 4, "l": 4, "hackel": [4, 7, 10], "tu": [4, 12], "berlin": [4, 12], "de": [4, 12], "complaint": 4, "review": 4, "investig": 4, "promptli": 4, "fairli": 4, "oblig": 4, "privaci": 4, "secur": 4, "incid": 4, "follow": [4, 5, 7, 8, 10, 11], "impact": 4, "consequ": 4, "violat": 4, "unprofession": 4, "unwelcom": 4, "written": 4, "clariti": 4, "around": [4, 5, 13, 14], "explan": 4, "why": 4, "apologi": 4, "singl": [4, 13, 14], "seri": 4, "continu": 4, "involv": 4, "unsolicit": 4, "period": 4, "time": [4, 5], "avoid": 4, "extern": 4, "term": 4, "seriou": 4, "sustain": 4, "sort": 4, "allow": [4, 11, 13, 14], "aggress": 4, "disparag": 4, "adapt": 4, "www": 4, "org": [4, 10, 12], "code_of_conduct": 4, "were": [4, 14], "inspir": 4, "mozilla": 4, "ladder": 4, "common": [4, 6], "faq": 4, "translat": 4, "project": [5, 10], "effort": [5, 10], "thank": 5, "you": [5, 9, 10, 11, 14, 15], "look": [5, 8], "how": [5, 8, 11], "There": [5, 11], "mani": [5, 8], "help": 5, "framework": [5, 10, 11, 13, 14], "document": [5, 7, 10, 11, 12], "grow": 5, "coupl": 5, "broader": 5, "easiest": 5, "improv": 5, "highlight": 5, "further": 5, "refin": 5, "would": [5, 11], "discuss": 5, "someth": 5, "featur": [5, 10], "feel": 5, "github": [5, 10, 11], "tell": 5, "star": 5, "poetri": 5, "develop": [5, 10], "download": [5, 11, 12], "instal": [5, 10], "clone": 5, "repositori": [5, 12], "git": [5, 12], "com": 5, "lhackel": [5, 10], "tub": [5, 10], "cd": 5, "replic": 5, "your": 5, "speed": 5, "hack": 5, "pr": 5, "suit": 5, "still": 5, "successfulli": 5, "regener": 5, "our": [5, 12, 13, 14], "back": 5, "quickli": 5, "hard": 5, "stick": 5, "until": 5, "tool": [6, 10], "13": 6, "numpi": [6, 13, 14], "24": 6, "timm": [6, 10, 14], "9": [6, 12], "formali": 6, "6": 6, "26": 6, "appdir": 6, "option": [6, 8, 13, 14], "matplotlib": 6, "scikit": 6, "bolt": 6, "post1": 6, "fvcore": 6, "post20221221": 6, "group": [6, 12], "psutil": 6, "dev": 6, "pytest": 6, "coverag": 6, "mock": 6, "furo": 6, "2022": 6, "03": 6, "23": 6, "myst": 6, "nb": 6, "17": 6, "sphinx": 6, "autobuild": 6, "2021": [6, 12], "14": 6, "sphinxcontrib": 6, "bibtex": 6, "preprocessor": 6, "jupyterlab": 6, "tensorboardx": 6, "mit": 7, "copyright": 7, "c": 7, "2023": [7, 10], "leonard": [7, 10], "wayn": 7, "herebi": 7, "grant": [7, 10], "charg": 7, "obtain": 7, "copi": 7, "softwar": [7, 10, 12], "associ": [7, 8], "deal": 7, "restrict": 7, "modifi": 7, "merg": 7, "sublicens": 7, "sell": 7, "permit": 7, "whom": 7, "furnish": 7, "do": [7, 13, 14], "subject": 7, "condit": 7, "abov": [7, 13, 14], "notic": 7, "shall": 7, "substanti": 7, "portion": 7, "THE": 7, "AS": 7, "warranti": 7, "OF": 7, "OR": 7, "impli": 7, "BUT": 7, "NOT": [7, 14], "TO": 7, "merchant": 7, "FOR": 7, "particular": 7, "purpos": 7, "AND": 7, "noninfring": 7, "IN": 7, "NO": 7, "author": [7, 10], "holder": 7, "BE": 7, "liabl": 7, "claim": 7, "damag": 7, "liabil": 7, "whether": 7, "contract": 7, "tort": 7, "aris": [7, 10], "connect": 7, "WITH": 7, "central": 8, "dataclass": 8, "ilmconfigur": [8, 13, 14], "decid": 8, "part": [8, 9, 15], "consist": 8, "task": [8, 11, 14], "ultim": 8, "solv": 8, "minim": [8, 10], "supervis": [8, 11], "properti": 8, "unus": 8, "type": 8, "fusion": 8, "ilmtyp": [8, 13, 14], "model_config": [8, 13, 14], "timm_model_nam": [8, 13, 14], "resnet18": [8, 13, 14], "seen": 8, "str": 8, "param": 8, "hf_model_nam": [8, 13, 14], "class_nam": 8, "network_typ": [8, 13, 14], "enum": 8, "vision_classif": [8, 13], "visual_features_out": 8, "512": 8, "fusion_in": 8, "fusion_out": 8, "fusion_hidden": 8, "256": 8, "v_dropout_r": 8, "float": [8, 13, 14], "t_dropout_r": 8, "fusion_dropout_r": 8, "fusion_method": 8, "callabl": 8, "mul": 8, "fusion_activ": 8, "tanh": 8, "drop_rat": 8, "use_pooler_output": 8, "bool": 8, "max_sequence_length": 8, "load_timm_if_avail": 8, "load_hf_if_avail": 8, "facilit": 8, "organ": 8, "code": [8, 12], "global": 8, "variabl": 8, "vqa_classif": [8, 14], "content": [9, 15], "select": [9, 13, 14, 15], "activ": [9, 10, 15], "beta": 10, "wip": [10, 15], "librari": [10, 12, 14], "state": 10, "art": 10, "seek": 10, "rapidli": 10, "iter": 10, "model": [10, 11], "sourc": [10, 11], "conveni": 10, "implement": 10, "seamlessli": 10, "popular": 10, "highli": 10, "regard": 10, "huggingfac": [10, 14], "With": 10, "extens": 10, "nearli": 10, "1000": [10, 14], "over": 10, "000": 10, "upload": 10, "offer": 10, "rang": [10, 13, 14], "Its": 10, "vast": 10, "unparallel": 10, "resourc": 10, "innov": 10, "sophist": 10, "furthermor": 10, "boast": 10, "friendli": 10, "interfac": 10, "streamlin": 10, "exchang": 10, "compon": 10, "thu": 10, "endless": 10, "novel": 10, "built": 10, "throughput": 10, "optim": [10, 13, 14], "r": 10, "moreov": 10, "comprehens": 10, "instruct": 10, "tutori": 10, "overview": 10, "ensur": 10, "smooth": 10, "hassl": 10, "outlin": 10, "process": [10, 12], "upcom": 10, "subsequ": 10, "explor": [10, 12], "exemplifi": 10, "encourag": 10, "visit": 10, "dedic": 10, "receiv": 10, "assist": 10, "submit": 10, "platform": 10, "cite": 10, "lhackel_tub_2023": 10, "kai": 10, "norman": 10, "clasen": 10, "beg\u00fcm": [10, 12], "demir": [10, 12], "titl": 10, "v0": 10, "month": 10, "apr": 10, "year": 10, "doi": [10, 12], "5281": 10, "7875406": 10, "url": [10, 12], "european": 10, "research": [10, 12], "council": 10, "erc": 10, "2017": 10, "stg": 10, "bigearth": 10, "under": 10, "759764": 10, "agenc": 10, "da4dt": 10, "precursor": 10, "digit": 10, "twin": 10, "earth": 10, "german": 10, "ministri": 10, "affair": 10, "climat": 10, "ai": 10, "cube": 10, "50ee2012b": 10, "pypi": 11, "pip": 11, "directli": 11, "want": 11, "wheel": 11, "equat": 11, "py3": 11, "whl": 11, "them": 11, "pretrain": [11, 14], "fashion": [11, 13, 14], "checkpoint": [11, 14], "explain": 11, "next": 11, "been": 11, "togeth": 11, "googl": 11, "colab": 11, "show": 11, "link": 12, "mm": 12, "nomenclatur": 12, "websit": 12, "everi": 12, "run": 12, "procedur": 12, "tensorflow": 12, "rsim": 12, "s2_43": 12, "classes_model": 12, "s2_19": 12, "simultan": 12, "mm_19": 12, "s1_tool": 12, "geotiff": 12, "script": [12, 13, 14], "extract": 12, "1c": 12, "grd": 12, "tile": 12, "disk": 12, "s2_tool": 12, "while": [12, 13, 14], "skip": 12, "cloudi": 12, "snowi": 12, "archiv": 12, "mm_tool": 12, "sylvain": 12, "beg": 12, "\u00fc": 12, "m": 12, "devi": 12, "tuia": 12, "rsvqa": 12, "meet": 12, "larg": 12, "scale": 12, "visual": 12, "ieee": 12, "intern": [12, 13, 14], "geoscienc": 12, "symposium": 12, "igarss": 12, "1218": 12, "1221": 12, "mengy": 12, "ryan": 12, "kiro": 12, "richard": 12, "zemel": 12, "neural": [12, 14], "system": 12, "2015": 12, "gencer": 12, "marcela": 12, "charfuelan": 12, "volker": 12, "markl": 12, "benchmark": 12, "understand": 12, "2019": 12, "juli": 12, "1109": 12, "8900532": 12, "arn": 12, "wall": 12, "tristan": 12, "kreuzig": 12, "filip": 12, "marcelino": 12, "hugo": 12, "costa": 12, "pedro": 12, "benevid": 12, "mario": 12, "caetano": 12, "multimod": 12, "retriev": 12, "geosci": 12, "sen": 12, "mag": 12, "174": 12, "180": 12, "septemb": 12, "mgr": 12, "3089174": 12, "trainer": [13, 14], "integr": [13, 14], "lightningmodul": [13, 14], "pl": [13, 14], "divid": [13, 14], "usual": [13, 14], "loop": [13, 14], "necessari": [13, 14], "training_step": [13, 14], "configure_optim": [13, 14], "fulli": [13, 14], "add": [13, 14], "step": [13, 14], "evalu": [13, 14], "_step": [13, 14], "_epoch_end": [13, 14], "litvisionencod": 13, "wrapper": [13, 14], "among": [13, 14], "thing": [13, 14], "gpu": [13, 14], "cpu": [13, 14], "def": [13, 14], "__init__": [13, 14], "self": [13, 14], "config": [13, 14], "lr": [13, 14], "1e": [13, 14], "super": [13, 14], "batch_idx": [13, 14], "x": [13, 14], "y": [13, 14], "x_hat": [13, 14], "loss": [13, 14], "binary_cross_entropy_with_logit": [13, 14], "log": [13, 14], "adamw": [13, 14], "weight_decai": [13, 14], "01": [13, 14], "mandatori": [13, 14], "validation_step": [13, 14], "validation_epoch_end": [13, 14], "avg_loss": [13, 14], "stack": [13, 14], "test_step": [13, 14], "test_epoch_end": [13, 14], "forward": [13, 14], "becaus": [13, 14], "inner": [13, 14], "manual": [13, 14], "tensorboard": [13, 14], "callback": [13, 14], "hyperparamet": [13, 14], "model_nam": [13, 14], "seed": [13, 14], "42": [13, 14], "number_of_channel": [13, 14], "epoch": [13, 14], "5e": [13, 14], "Then": [13, 14], "later": [13, 14], "random": [13, 14], "spawn": [13, 14], "subprocess": [13, 14], "seed_everyth": [13, 14], "max_epoch": [13, 14], "acceler": [13, 14], "log_every_n_step": [13, 14], "logger": [13, 14], "final": [13, 14], "bendatamodul": 13, "quit": [13, 14], "bit": [13, 14], "readabl": [13, 14], "sinc": [13, 14], "color": [13, 14], "slightli": [13, 14], "distort": [13, 14], "anywai": [13, 14], "real": [13, 14], "8461952209472656": 13, "053515881299972534": 13, "43593648076057434": 13, "5633976459503174": 13, "5838469862937927": 13, "5952039957046509": 13, "7836412191390991": 13, "726469874382019": 13, "7546876072883606": 13, "8279280662536621": 13, "19324178993701935": 13, "7901748418807983": 13, "6696745753288269": 13, "8135374784469604": 13, "9674454927444458": 13, "7233631610870361": 13, "6134014129638672": 13, "8439663648605347": 13, "7069821953773499": 13, "both": 14, "either": 14, "weight": 14, "composit": 14, "rsvqaxben": 14, "instead": 14, "_disassemble_batch": 14, "disassembl": 14, "litvqaencod": 14, "transpos": 14, "tolist": 14, "t": 14, "image_model_nam": 14, "text_model_nam": 14, "warn": 14, "known": 14, "keyword": 14, "convolut": 14, "cnn": 14, "resnet": 14, "oper": 14, "independ": 14, "rsvqaxbendatamodul": 14, "131": 14, "userwarn": 14, "unknown": 14, "ignor": 14, "restart": 14, "cach": 14, "pretrained_model": 14, "huggingface_model": 14, "bertmodel": 14, "classifi": 14, "bia": 14, "anoth": 14, "architectur": 14, "bertforsequenceclassif": 14, "bertforpretrain": 14, "exactli": 14, "115": 14, "again": 14, "heterogen": 14, "1998": 14, "21770": 14, "10624": 14, "6914": 14, "14769": 14, "24331672489643097": 14, "416130393743515": 14, "4097263514995575": 14, "5235225558280945": 14, "2558377981185913": 14, "4784601926803589": 14, "32770946621894836": 14, "9055449366569519": 14, "7889048457145691": 14, "7592024207115173": 14}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"us": 0, "bigearthnet": [0, 1, 12], "lmdb": 0, "reader": 0, "select": [0, 1, 3], "band": [0, 1, 3], "label": 0, "type": 0, "mean": 0, "standard": [0, 4], "deviat": 0, "dataset": [1, 2, 3, 13, 14], "datamodul": [1, 3], "bendataset": 1, "split": [1, 3], "restrict": [1, 3], "number": [1, 3], "load": [1, 3], "imag": [1, 3, 9, 13], "bendatamodul": 1, "dataload": [1, 3], "set": [1, 3], "coco": 2, "qa": 2, "cocoqa": 2, "rsvqaxben": 3, "rsvqaxbendataset": 3, "token": 3, "class": 3, "specif": 3, "answer": [3, 14], "rsvqaxbendatamodul": 3, "contributor": 4, "coven": 4, "code": [4, 5], "conduct": 4, "our": 4, "pledg": 4, "enforc": 4, "respons": 4, "scope": 4, "guidelin": 4, "1": 4, "correct": 4, "2": 4, "warn": 4, "3": 4, "temporari": 4, "ban": 4, "4": 4, "perman": 4, "attribut": 4, "contribut": 5, "give": 5, "feedback": 5, "increas": 5, "visibl": 5, "directli": 5, "updat": 5, "sourc": 5, "notebook": 5, "depend": 6, "python": 6, "poetri": 6, "licens": 7, "model": [8, 12, 13, 14], "configur": [8, 13, 14], "wip": 9, "caption": 9, "configilm": 10, "instal": 11, "further": 12, "refer": 12, "The": 12, "guid": 12, "pretrain": 12, "tool": 12, "bibliographi": 12, "supervis": 13, "classif": 13, "pytorch": [13, 14], "lightn": [13, 14], "modul": [13, 14], "creat": [13, 14], "run": [13, 14], "visual": 14, "question": 14, "vqa": 14}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 57}, "alltitles": {"Using the BigEarthNet LMDB Reader": [[0, "using-the-bigearthnet-lmdb-reader"]], "Selecting Bands": [[0, "selecting-bands"], [1, "selecting-bands"], [3, "selecting-bands"]], "Label types": [[0, "label-types"]], "Mean and Standard Deviation": [[0, "mean-and-standard-deviation"]], "BigEarthNet Dataset & Datamodule": [[1, "bigearthnet-dataset-datamodule"]], "BENDataSet": [[1, "bendataset"]], "Splits": [[1, "splits"], [3, "splits"]], "Restricting the number of loaded images": [[1, "restricting-the-number-of-loaded-images"], [3, "restricting-the-number-of-loaded-images"]], "BENDataModule": [[1, "bendatamodule"]], "DataLoader settings": [[1, "dataloader-settings"], [3, "dataloader-settings"]], "COCO-QA": [[2, "coco-qa"]], "COCOQA DataSet": [[2, "cocoqa-dataset"]], "RSVQAxBEN Dataset & Datamodule": [[3, "rsvqaxben-dataset-datamodule"]], "RSVQAxBENDataSet": [[3, "rsvqaxbendataset"]], "Tokenizer and Tokenization": [[3, "tokenizer-and-tokenization"]], "Select Number of Classes or specific Answers": [[3, "select-number-of-classes-or-specific-answers"]], "RSVQAxBENDataModule": [[3, "rsvqaxbendatamodule"]], "Contributor Covenant Code of Conduct": [[4, "contributor-covenant-code-of-conduct"]], "Our Pledge": [[4, "our-pledge"]], "Our Standards": [[4, "our-standards"]], "Enforcement Responsibilities": [[4, "enforcement-responsibilities"]], "Scope": [[4, "scope"]], "Enforcement": [[4, "enforcement"]], "Enforcement Guidelines": [[4, "enforcement-guidelines"]], "1. Correction": [[4, "correction"]], "2. Warning": [[4, "warning"]], "3. Temporary Ban": [[4, "temporary-ban"]], "4. Permanent Ban": [[4, "permanent-ban"]], "Attribution": [[4, "attribution"]], "Contributing": [[5, "contributing"]], "Give feedback": [[5, "give-feedback"]], "Increasing visibility": [[5, "increasing-visibility"]], "Directly update source code or notebooks": [[5, "directly-update-source-code-or-notebooks"]], "Dependencies": [[6, "dependencies"]], "Python (Poetry) Dependencies": [[6, "python-poetry-dependencies"]], "License": [[7, "license"]], "Model Configuration": [[8, "model-configuration"]], "[WIP] Image Captioning": [[9, "wip-image-captioning"]], "ConfigILM": [[10, "configilm"]], "Installation": [[11, "installation"]], "Further references": [[12, "further-references"]], "The BigEarthNet Guide": [[12, "the-bigearthnet-guide"]], "Pretrained models": [[12, "pretrained-models"]], "BigEarthNet Tools": [[12, "bigearthnet-tools"]], "Bibliography": [[12, "bibliography"]], "Supervised Image Classification": [[13, "supervised-image-classification"]], "Pytorch Lightning Module": [[13, "pytorch-lightning-module"], [14, "pytorch-lightning-module"]], "Configuring": [[13, "configuring"], [14, "configuring"]], "Creating Model + Dataset": [[13, "creating-model-dataset"], [14, "creating-model-dataset"]], "Running": [[13, "running"], [14, "running"]], "Visual Question Answering (VQA)": [[14, "visual-question-answering-vqa"]]}, "indexentries": {}})