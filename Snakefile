EXPS = [
    "agpt_benchmark_alex",
    "agpt_benchmark_jarvis",
    "cdvae_benchmark_alex",
    "cdvae_benchmark_jarvis",
    "flowmm_benchmark_alex",
    "flowmm_benchmark_jarvis",
]

for exp in EXPS:
    module:
        name: exp
        snakefile: f"job_runs/{exp}/Snakefile"
    use rule * from exp

# Ensure the sentinels directory exists
rule make_sentinels_dir:
    output:
        directory("sentinels")
    shell:
        "mkdir -p {output}"

rule all:
    input:
        expand("{exp}.final", exp=EXPS),
        "sentinels/charts.made",
        "sentinels/overlay_charts.created",
        "sentinels/benchmarks.verified",
        "sentinels/grid_charts.created"

rule make_atomgpt_env:
    input:
        directory("sentinels")
    output:
        touch("sentinels/atomgpt_env.created")
    shell:
        """
        bash job_runs/agpt_benchmark_alex/conda_env.job
        """

rule make_cdvae_env:
    input:
        directory("sentinels")
    output:
        touch("sentinels/cdvae_env.created")
    shell:
        """
        bash job_runs/cdvae_benchmark_alex/conda_env.job
        """

rule make_flowmm_env:
    input:
        directory("sentinels")
    output:
        touch("sentinels/flowmm_env.created")
    shell:
        """
        bash job_runs/flowmm_benchmark_alex/conda_env.job
        """

rule envs_ready:
    input:
        "sentinels/atomgpt_env.created",
        "sentinels/cdvae_env.created",
        "sentinels/flowmm_env.created"
    output:
        touch("sentinels/all_envs_ready.txt")
    shell:
        """
        echo 'all conda envs ready' > {output}
        """

rule make_jarvis_data:
    input:
        "sentinels/all_envs_ready.txt"
    output:
        touch("sentinels/jarvis_data.created")
    shell:
        """
        dvc --cd tc_supercon repro
        """

rule make_alex_data:
    input:
        "sentinels/all_envs_ready.txt"
    output:
        touch("sentinels/alex_data.created")
    shell:
        """
        dvc --cd alexandria repro
        """

rule make_stats_yamls:
    input:
        "sentinels/flowmm_env.created",
        "sentinels/jarvis_data.created",
        "sentinels/alex_data.created"
    output:
        touch("sentinels/flowmm_yamls.created")
    shell:
        """
        bash job_runs/flowmm_benchmark_alex/yamls.sh
        """

rule compile_results:
    input:
        expand("{exp}.final", exp=EXPS),
    output:
        touch("sentinels/metrics.computed")
    shell:
        """
        cd job_runs/ && bash ../scripts/loop.sh
        """

rule verify_benchmarks:
    input:
        "sentinels/metrics.computed"
    output:
        touch("sentinels/benchmarks.verified")
    shell:
        """
        python scripts/verify_benchmarks.py --root job_runs/
        """

rule make_bar_charts:
    input:
        "sentinels/metrics.computed"
    output:
        touch("sentinels/charts.made")
    shell:
        "cd job_runs/ && python ../scripts/bar_chart.py"

rule make_overlay_charts:
    input:
        "sentinels/alex_data.created",
        "sentinels/jarvis_data.created"
    output:
        touch("sentinels/overlay_charts.created")
    shell:
        """
        bash scripts/make_overlay_charts.sh
        """

rule make_grid_charts:
    input:
        "sentinels/metrics.computed"
    output:
        touch("sentinels/grid_charts.created")
    shell:
        """
        python scripts/grid_charts.py
        """

