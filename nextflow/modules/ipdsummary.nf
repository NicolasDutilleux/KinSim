// ipdSummary — two backends selectable via params.ipd_backend:
//
//   'module'    → SMRT-Link 12.0 CLI + SP3-C3 null model (kineticsTools 12).
//                 Matches the Vega / Strepto bash pipelines.
//   'container' → pacbio-smrt-tools-25.3.sif default model.
//                 Matches the Sequel subread pipeline.
//
// All tool flags (identify, methylFraction, minCoverage, pvalue, arbitrary extras)
// are routed through params.ipd_* — edit YAML, don't edit code.

process IPDSUMMARY {
    tag "${sample_id}"
    label 'heavy'
    storeDir "${params.outdir}/${sample_id}"

    input:
    tuple val(sample_id), path(aligned_bam), path(bai), path(pbi), path(reference)

    output:
    tuple val(sample_id),
          path("${sample_id}_ipdSummary.gff"),
          path("${sample_id}_ipdSummary.csv"),
          path(aligned_bam),
          path(reference)

    script:
    def identify    = params.ipd_identify ?: 'm6A,m4C'
    def methyl_frac = params.ipd_methyl_fraction ? '--methylFraction' : ''
    def min_cov     = params.ipd_min_coverage ? "--minCoverage ${params.ipd_min_coverage}" : ''
    def pvalue      = params.ipd_pvalue ? "--pvalue ${params.ipd_pvalue}" : ''
    def extra       = params.ipd_extra ?: ''

    def flags = [
        "--reference ${reference}",
        "--identify ${identify}",
        methyl_frac,
        min_cov,
        pvalue,
        "--csv ${sample_id}_ipdSummary.csv",
        "--gff ${sample_id}_ipdSummary.gff",
        "--numWorkers ${task.cpus}",
        extra
    ].findAll { it }.join(' ')

    if (params.ipd_backend == 'module') {
        def model_flag = "--ipdModel ${params.ipd_ipd_model ?: params.sp3_model}"
        """
        source /etc/profile.d/modules.sh 2>/dev/null || true
        module load ${params.smrtlink_mod} 2>/dev/null || true

        if [ ! -f ${reference}.fai ]; then
            samtools faidx ${reference}
        fi

        ipdSummary ${aligned_bam} ${flags} ${model_flag}
        """
    } else if (params.ipd_backend == 'container') {
        """
        if [ ! -f ${reference}.fai ]; then
            samtools faidx ${reference}
        fi

        apptainer exec --bind /data ${params.smrt_sif} \\
            ipdSummary ${aligned_bam} ${flags}
        """
    } else {
        error "Unknown ipd_backend: '${params.ipd_backend}' (expected 'module' or 'container')"
    }
}
