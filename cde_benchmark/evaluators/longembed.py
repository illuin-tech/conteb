from cde_benchmark.formatters.data_formatter import LongEmbedDataFormatter

DATASETS = ["summ_screen_fd", "narrativeqa", "2wikimqa", "qmsum", "needle", "passkey"]

MULTI_CONTEXT_TASKS = ["needle", "passkey"]
CONTEXT_LENGTHS = [
    32768,
    16384,
    8192,
    4096,
    2048,
    1024,
    512,
    256,
]


class LongEmbedEvaluator:
    def __init__(
        self,
        path,
        embedder,
        query_prompt="",
        doc_prompt="",
        datasets=DATASETS,
        multi_context_tasks=MULTI_CONTEXT_TASKS,
        context_lengths=CONTEXT_LENGTHS,
    ):
        self.datasets = datasets
        self.path = path
        self.embedder = embedder
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self.multi_context_tasks = multi_context_tasks
        self.context_lengths = context_lengths

    def run_multi_context_task(self, task):
        all_metrics = {}
        avg_metrics = {}
        for context_length in self.context_lengths:
            print(f"Running task {task} with context length: {context_length}")
            formatter = LongEmbedDataFormatter(
                self.path,
                task,
                query_prompt=self.query_prompt,
                doc_prompt=self.doc_prompt,
                context_length=str(context_length),
            )
            metrics = self.embedder.compute_metrics_e2e(formatter)
            for k, v in metrics.items():
                print(f"\t{k}: {v}")
                if k not in avg_metrics:
                    avg_metrics[k] = []
                avg_metrics[k].append(v)
            print("=" * 20)

            all_metrics[f"{task}_{context_length}"] = metrics

        # average metrics across context lengths
        for k, v in avg_metrics.items():
            avg_metrics[k] = sum(v) / len(v)
            print(f"\tAverage {k}: {avg_metrics[k]}")

        all_metrics[f"{task}_avg"] = avg_metrics

        return all_metrics

    def run_task(self, task):
        if task in self.multi_context_tasks:
            return self.run_multi_context_task(task)

        formatter = LongEmbedDataFormatter(self.path, task, query_prompt=self.query_prompt, doc_prompt=self.doc_prompt)
        metrics = self.embedder.compute_metrics_e2e(formatter)
        return metrics

    def run_all_tasks(self):
        results = {}
        for task in self.datasets:
            print(f"Running task: {task}")
            res = self.run_task(task)
            print(res)
            results[task] = res
        return results
