// De novo HiFi assembly. Runs only when a sample has no reference column.
// Tunable via params.hifiasm_extra (arbitrary extra CLI args).

process HIFIASM {
    tag "${sample_id}"
    label 'assembly'
    storeDir "${params.outdir}/${sample_id}"

    input:
    tuple val(sample_id), path(hifi_bam)

    output:
    tuple val(sample_id), path("${sample_id}_assembly.fasta")

    script:
    def extra = params.hifiasm_extra ?: ''
    """
    samtools fastq ${hifi_bam} | gzip > ${sample_id}.fastq.gz

    hifiasm -o ${sample_id}_asm -t ${task.cpus} ${extra} ${sample_id}.fastq.gz

    awk '/^S/{print ">"\$2; print \$3}' \\
        ${sample_id}_asm.bp.p_ctg.gfa > ${sample_id}_assembly.fasta

    samtools faidx ${sample_id}_assembly.fasta

    rm -f ${sample_id}_asm.*.bin ${sample_id}.fastq.gz
    """
}
