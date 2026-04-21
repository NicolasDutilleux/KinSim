// samtools index + pbindex — ipdSummary needs both.

process INDEX_BAM {
    tag "${sample_id}"
    label 'small'
    storeDir "${params.outdir}/${sample_id}"

    input:
    tuple val(sample_id), path(aligned_bam), path(reference)

    output:
    tuple val(sample_id),
          path(aligned_bam),
          path("${aligned_bam}.bai"),
          path("${aligned_bam}.pbi"),
          path(reference)

    script:
    """
    if [ ! -f ${aligned_bam}.bai ]; then
        samtools index ${aligned_bam}
    fi
    if [ ! -f ${aligned_bam}.pbi ]; then
        apptainer exec --bind /data ${params.smrt_sif} pbindex ${aligned_bam}
    fi
    """
}
