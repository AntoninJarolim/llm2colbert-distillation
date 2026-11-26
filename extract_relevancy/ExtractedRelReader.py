class ExtractedRelevancyReader:
    def __init__(self, path):
        self.path = path
        self.lines = open(path, encoding="utf-8").read().splitlines()

    def __iter__(self):
        for line in self.lines:
            parts = line.split("\t")
            query_id, passage_id, t, *spans = parts
            yield query_id, passage_id, t, spans