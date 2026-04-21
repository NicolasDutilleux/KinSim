// Decompress `.bam.gz` inputs (e.g. Sequel II's lima.bcXXXX--bcXXXX.bam.gz).

process GUNZIP_BAM {
    tag "${sample_id}"
    label 'small'
    storeDir "${params.outdir}/${sample_id}"

    input:
    tuple val(sample_id), path(gz)

    output:
    tuple val(sample_id), path("${sample_id}_reads.bam")

    script:
    """
    gunzip -c ${gz} > ${sample_id}_reads.bam
    """
}
