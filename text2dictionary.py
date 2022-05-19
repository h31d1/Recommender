import re

def collect_objects(filename):
    collect = []
    obj = {"Id": "","ASIN": "","title": "","group": "","salesrank": "","no_sim": "",
        "similar": [],"categories": "","catlists": [],"reviews": "","revdicts": []}
    keys = ["Id","ASIN","title","group","salesrank","similar","categories","reviews"]
    i = 0
    with open(filename, encoding="utf8") as f:
        for line in f:
            i += 1
            line = line.strip() #.replace("'","").replace('"','').replace("\\","")
            if line == "":
                collect.append(obj)
                obj = {"Id": "","ASIN": "","title": "","group": "","salesrank": "","no_sim": "",
                    "similar": [],"categories": "","catlists": [],"reviews": "","revdicts": []}
            key = line.split(":")[0]
            if key in keys:
                obj[key] = line[(len(key)+2):].strip()
                if key in ["Id","salesrank","categories"]:
                    value = obj[key]
                    obj[key] = int(value)
                if key == "similar":
                    value = obj[key].split()
                    obj[key] = value[1:]
                    obj["no_sim"] = int(value[0])
                if key == "reviews":
                    value = obj[key].split("  ")
                    obj[key] = {}
                    for v in value:
                        obj[key][v.split(":")[0].strip()] = float(v.split(":")[1].strip())
            if (len(line)>0) and (key not in keys):
                if line[0] == "|":
                    obj["catlists"].append(line[1:].split("|"))
                if re.match(r"(\d{4}-)", line) != None:
                    splits = line.split()
                    if len(splits) == 9:
                        revdict = {
                            "date":splits[0],
                            "customer":splits[2],
                            "rating":float(splits[4]),
                            "votes":float(splits[6]),
                            "helpful":float(splits[8])
                        }
                        obj["revdicts"].append(revdict)
    return collect