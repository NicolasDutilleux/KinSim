// ccs-kinetics-bystrandify — unfold HiFi CCS into per-strand records with ip/pw.
// Only runs when platform == 'ccs'.  Inline `apptainer exec --bind /data` matches
// the existing bash pipeline exactly and avoids Nextflow-managed container quirks.

process BYSTRANDIFY {
    tag "${sample_id}"
    label 'heavy'
    storeDir "${params.outdir}/${sample_id}"

    input:
    tuple val(sample_id), path(hifi_bam)

    output:
    tuple val(sample_id), path("${sample_id}_bystrandify.bam")

    script:
    """
    apptainer exec --bind /data ${params.smrt_sif} \\
        ccs-kinetics-bystrandify ${hifi_bam} ${sample_id}_bystrandify.bam
    """
}
