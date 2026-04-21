#!/usr/bin/env nextflow
// ============================================================
// KinSim PREPARE pipeline — entrypoint.
//
// A single generic workflow (PREPARE) handles any HiFi / subread dataset.
// The pipeline shape is controlled by params (see nextflow.config). Per-dataset
// defaults live in conf/<name>.config and are activated via profiles.
//
//   Reads (.bam | .bam.gz) → [gunzip] → [bystrandify if ccs] →
//   pbmm2 align → samtools index + pbindex → ipdSummary → manifest_gff.csv
//
// Usage:
//   nextflow run nextflow/main.nf --help
//
//   nextflow run nextflow/main.nf \\
//       -profile vega,slurm \\
//       -params-file nextflow/params/vega.yaml \\
//       -resume
//
// Adding a new dataset:
//   1. Create nextflow/samplesheets/<name>.csv  (sample_id,reads[,reference])
//   2. Copy nextflow/params/vega.yaml → nextflow/params/<name>.yaml and tweak
//   3. (Optional) Create conf/<name>.config + a named profile if defaults differ
//   4. nextflow run nextflow/main.nf -profile slurm -params-file nextflow/params/<name>.yaml
// ============================================================

nextflow.enable.dsl = 2

include { PREPARE } from './workflows/prepare.nf'

def printHelp() {
    log.info """
    ================================================================
     KinSim PREPARE pipeline (v${workflow.manifest.version})
    ================================================================

    nextflow run nextflow/main.nf -profile <dataset>[,slurm] \\
        -params-file nextflow/params/<dataset>.yaml [options]

    REQUIRED (via -params-file or CLI)
      --samplesheet   CSV: sample_id,reads[,reference]
      --outdir        per-sample output base dir

    PIPELINE SHAPE
      --platform      'ccs' | 'subread'       (default: ccs)
      --assemble      true|false              (default: false — hifiasm if ref empty)

    TOOL TUNING
      --hifiasm_extra       extra hifiasm CLI flags
      --pbmm2_preset        override (else derived from platform)
      --pbmm2_sort          default: true
      --pbmm2_extra         extra pbmm2 CLI flags
      --ipd_backend         'module' | 'container'       (default: module)
      --ipd_identify        e.g. 'm6A,m4C'               (default: m6A,m4C)
      --ipd_methyl_fraction true|false
      --ipd_min_coverage    int
      --ipd_pvalue          float
      --ipd_ipd_model       override SP3-C3 path         (module backend)
      --ipd_extra           extra ipdSummary CLI flags

    PROFILES
      -profile local                 run on current machine
      -profile slurm                 submit to SLURM
      -profile vega|strepto|sequel   apply dataset preset (combine with slurm/local)
      -profile debug                 verbose, no cache

    SKIP LOGIC
      Every process uses storeDir=<outdir>/<sample_id>/. If all its declared
      outputs exist there, the process is skipped. Use -resume for the work-dir
      cache on top of that. To rerun one step for one sample, delete its file.

    OUTPUT
      <outdir>/<sample_id>/<sid>_aligned.bam[.bai][.pbi]
      <outdir>/<sample_id>/<sid>_ipdSummary.{gff,csv}
      <outdir>/manifest_gff.csv  — consumed by slurm_kinsim/00_extract.slurm
    """.stripIndent()
}

if (params.help) {
    printHelp()
    exit 0
}

def missing = []
if (!params.samplesheet) missing << '--samplesheet'
if (!params.outdir)      missing << '--outdir'
if (missing) {
    log.error "Missing required parameter(s): ${missing.join(', ')}\n"
    printHelp()
    exit 1
}

if (!(params.platform in ['ccs', 'subread'])) {
    error "Unknown --platform '${params.platform}' (must be 'ccs' or 'subread')"
}
if (!(params.ipd_backend in ['module', 'container'])) {
    error "Unknown --ipd_backend '${params.ipd_backend}' (must be 'module' or 'container')"
}
if (!file(params.samplesheet).exists()) {
    error "Samplesheet not found: ${params.samplesheet}"
}

log.info """
================================================================
 KinSim PREPARE v${workflow.manifest.version}
================================================================
 samplesheet  : ${params.samplesheet}
 outdir       : ${params.outdir}
 platform     : ${params.platform}
 assemble     : ${params.assemble}
 ipd_backend  : ${params.ipd_backend}
 profile      : ${workflow.profile ?: 'default'}
 resume       : ${workflow.resume}
================================================================
""".stripIndent()

workflow {
    PREPARE()
}

workflow.onComplete {
    def status = workflow.success ? 'COMPLETED' : 'FAILED'
    log.info "\nPipeline ${status} in ${workflow.duration}"
    if (workflow.success) {
        def mname = params.manifest_name ?: 'manifest_gff.csv'
        log.info "Manifest : ${params.outdir}/${mname}"
        log.info "Reports  : ${params.tracedir ?: params.outdir + '/_nextflow'}/"
        log.info ""
        log.info "Next step (extract + merge):"
        log.info "  sbatch slurm_kinsim/00_extract.slurm ${params.outdir}/${mname}"
    }
}
