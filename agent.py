# Define a memory-enabled CodeAgent
class SimpleCodeAgentWithMemory:
    def __init__(self, model, system_prompt: str, max_iterations: int = 5):
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.memory = []

    def add_to_memory(self, role: str, content: str):
        self.memory.append({"role": role, "content": content})

    def run(self, task: str) -> str:
        self.add_to_memory("user", task)

        for iteration in range(self.max_iterations):
            try:
                context = [{"role": "system", "content": self.system_prompt}] + self.memory
                response = self.model(context)
                llm_output = response["choices"][0]["message"]["content"]
                self.add_to_memory("assistant", llm_output)

                code = self._extract_code(llm_output)
                if not code:
                    return f"Agent completed: {llm_output.strip()}"

                observation = self._execute_code(code)
                self.add_to_memory("assistant", f"Observation:\n{observation}")
                return observation

            except Exception as e:
                error_message = f"Error during execution: {e}"
                self.add_to_memory("assistant", error_message)
                return error_message

        return "Max iterations reached without a final result."

    @staticmethod
    def _extract_code(llm_output: str) -> str:
        import re
        match = re.search(r"```python(.*?)```", llm_output, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _execute_code(code: str) -> str:
        safe_globals = {}
        safe_locals = {}

        try:
            exec(code, safe_globals, safe_locals)
            return safe_locals.get("result", "Code executed successfully, but no 'result' was returned.")
        except Exception as e:
            return f"Execution error: {e}"