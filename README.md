# zerovram

Run a private, offline coding agent on a laptop with **no graphics card**, just the Intel Iris Xe / Arc
integrated GPU.

## The one-line finding

Ollama detects Intel integrated GPUs and then **ignores them on purpose**. The server log says:

```
dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1
```

Set two environment variables and the same model runs on the iGPU instead of the CPU:

```
OLLAMA_VULKAN=1
OLLAMA_IGPU_ENABLE=1
```

## Architecture

```mermaid
flowchart TB
    subgraph You["You (terminal)"]
        U[Prompt]
    end

    subgraph Agents["Agent layer (pick one)"]
        TA["tinyagent.py<br/>~300-token system prompt<br/>explores on its own"]
        AD["Aider<br/>edits files you /add"]
    end

    subgraph Tools["tinyagent tools (sandboxed to the current folder)"]
        R["list_dir · find_files<br/>grep · read_file"]
        W["write_file · run"]
        G{"You approve?<br/>y / N"}
    end

    subgraph Server["Ollama server"]
        API["/api/chat<br/>streaming + tool calls"]
        ENV["OLLAMA_VULKAN=1<br/>OLLAMA_IGPU_ENABLE=1"]
        LC["llama.cpp runner<br/>Vulkan backend"]
    end

    subgraph HW["Laptop hardware (no discrete GPU)"]
        IGPU["Intel Iris Xe iGPU<br/>runs the model"]
        CPU["CPU<br/>agent + OS"]
        RAM[("Shared DDR5 RAM<br/>model weights<br/>~30 GB/s measured")]
    end

    U --> TA & AD
    TA -- "messages + tool schema" --> API
    AD -- messages --> API
    API -- "streamed tokens / tool calls" --> TA
    TA --> R
    TA --> W --> G
    R -- "results (capped 6k chars)" --> TA
    G -- "yes" --> TA
    ENV -. "unlocks iGPU" .-> LC
    API --> LC --> IGPU
    IGPU <--> RAM
    CPU <--> RAM
```

**Flow:** you ask a question. The agent sends it to Ollama with a small tool list. The model runs on the iGPU and
either answers or asks for a tool. Read tools run immediately. Write and run tools wait for your `y`. The agent
loops until the model answers. The speed limit is the shared RAM: the iGPU and CPU read the same ~30 GB/s.

## Results

Test machine: i7-12700H, Iris Xe (96 EU), 40 GB DDR5-4800, no discrete GPU, Windows 11, Ollama 0.40.

Prompt: 3,480 tokens. Output: up to 128 tokens.

![CPU vs Iris Xe on qwen3-coder:30b](docs/benchmark.png)

| Model | Device | Prompt reading | Generation |
|---|---|---|---|
| qwen3:4b | CPU | 51.5 tok/s | 6.1 tok/s |
| qwen3:4b | **Iris Xe** | **143.1 tok/s (2.8x)** | **8.5 tok/s (1.4x)** |
| qwen3-coder:30b | CPU | 37.5 tok/s | 5.9 tok/s |
| qwen3-coder:30b | **Iris Xe** | **67.0 tok/s (1.8x)** | **9.8 tok/s (1.7x)** |

`qwen3-coder:30b` is a mixture-of-experts model: 30B parameters total, about 3B used per token.
That is why a 30B model is practical on a laptop at all.

### Why generation tops out near 10 tok/s

Generating a token means reading the model's active weights from memory, so memory bandwidth sets the speed limit.
The iGPU shares system RAM with the CPU. A custom OpenCL read kernel on the iGPU and a 20-thread CPU read both
stopped at about 30 GB/s:

![Measured memory bandwidth: CPU and iGPU both near 30 GB/s](docs/bandwidth.png)

On this machine one RAM module is soldered (8 GB) and one is a 32 GB stick. The mismatch puts most of the memory in
single-channel mode. Laptops with matched dual-channel RAM should do better. A discrete GPU reads its own VRAM at
300–1000 GB/s.

**Agent test:** [Aider](https://aider.chat) with `qwen3-coder:30b` on the Iris Xe fixed a two-bug Python file
so that its tests passed. It took 85 seconds end to end, including the cold model load.

## Setup (Windows)

You need [Ollama](https://ollama.com/download), [uv](https://docs.astral.sh/uv/), an up-to-date Intel graphics
driver, and 16 GB+ RAM. For the 30B model you need 36 GB+.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

The script does five things:

1. It picks a model that fits your RAM.
2. It sets the two variables above, plus Aider's model setting.
3. It restarts Ollama.
4. It checks the server log to confirm that Ollama detected the iGPU.
5. It pulls the model and installs Aider.

Then open a new terminal in your project and run `aider`.

To use a specific model, run `setup.ps1 -Model qwen2.5-coder:7b`.

## Setup (Linux, untested)

Set the same variables on the Ollama service, then restart it:

```bash
sudo systemctl edit ollama   # add: [Service]
                             #      Environment="OLLAMA_VULKAN=1" "OLLAMA_IGPU_ENABLE=1"
sudo systemctl restart ollama
journalctl -u ollama | grep "inference compute"   # should show type=iGPU
```

## tinyagent: an agent that explores by itself

Aider only sees files that you `/add`. `tinyagent.py` is a Claude-Code-style agent instead. It lists, finds,
greps and reads files without being told which ones. It asks you before it writes a file or runs a command.

Its system prompt is about 300 tokens, so each step costs seconds on an iGPU, not minutes. It needs only the Python
standard library.

```
python tinyagent.py                       # interactive, in the current folder
python tinyagent.py -p "summarize the design notes"
python test_tinyagent.py                  # sandbox + tool tests, no Ollama needed
```

On the test machine it answered a "find and summarize" question about a large folder of markdown notes in about 4 minutes.
It made 6–7 tool calls, ran 100% on the GPU, and used a 16k context. While it works it shows a live timer for each
step, streams the answer as it is generated, and prints every tool call.

qwen3-coder sometimes writes a tool call as plain text (`<function=...>`) instead of a structured call.
tinyagent detects that and runs the call anyway.

## Benchmark your own machine

```
python bench.py qwen3-coder:30b "iGPU"
python bench.py qwen3-coder:30b "CPU" 0     # num_gpu=0 forces CPU, for comparison
```

PRs with results from other chips are welcome: Arc iGPUs, Meteor Lake, Lunar Lake, and AMD Radeon iGPUs.

## Why Aider?

The prompt-reading speed is the weak point: about 67 tok/s on Iris Xe.

- Aider sends about 1–3k tokens per request, so the first answer arrives in seconds.
- Agents with very large system prompts (20k+ tokens) can wait several minutes before the first reply.

## Limits

- **Generation is limited by memory bandwidth.** The iGPU shares system RAM, so expect about 10 tok/s on a
  30B MoE model. A discrete GPU with its own VRAM is about 10x faster.
- **The iGPU can use about half of system RAM.** Large contexts can push part of the model back to the CPU.
  Run `ollama ps` to check: `100% GPU` is what you want.
- **Desktop responsiveness drops while a model runs**, because the display and the model share the same chip.
- **Model quality:** a local 30B model is strong for focused edits. It is not a frontier model.

## License

MIT
