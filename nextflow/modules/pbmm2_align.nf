// pbmm2 align. Preset is computed by the workflow from params.platform (or set
// explicitly via params.pbmm2_preset). Extra flags via params.pbmm2_extra.

process PBMM2_ALIGN {
    tag "${sample_id}"
    label 'heavy'
    storeDir "${params.outdir}/${sample_id}"

    input:
    tuple val(sample_id), path(reads_bam), path(reference), val(preset)

    output:
    tuple val(sample_id), path("${sample_id}_aligned.bam"), path(reference)

    script:
    def sort_flag = params.pbmm2_sort == false ? '' : '--sort'
    def extra     = params.pbmm2_extra ?: ''
    """
    if [ ! -f ${reference}.fai ]; then
        samtools faidx ${reference}
    fi

    pbmm2 align ${reference} ${reads_bam} ${sample_id}_aligned.bam \\
        --preset ${preset} ${sort_flag} \\
        --num-threads ${task.cpus} ${extra}
    """
}
