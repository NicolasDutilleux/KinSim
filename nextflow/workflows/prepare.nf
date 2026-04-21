// ============================================================
// PREPARE — generic per-sample preparation.
//
// Samplesheet (CSV with header):
//   sample_id,reads[,reference]
//
//   reads       path to BAM or BAM.GZ
//   reference   optional path to reference FASTA. If empty AND params.assemble,
//               hifiasm is run de novo from the reads.
//
// Output per sample (<outdir>/<sample_id>/):
//   <sid>_reads.bam            (only when input was .bam.gz)
//   <sid>_assembly.fasta       (only when assemble=true and no reference provided)
//   <sid>_bystrandify.bam      (only when platform=ccs)
//   <sid>_aligned.bam[.bai][.pbi]
//   <sid>_ipdSummary.gff
//   <sid>_ipdSummary.csv
//
// Gathered output:
//   <outdir>/manifest_gff.csv  (sample_id,bam_path,motifs,gff) — consumed by
//                              slurm_kinsim/00_extract.slurm
// ============================================================

include { HIFIASM }     from '../modules/hifiasm.nf'
include { BYSTRANDIFY } from '../modules/bystrandify.nf'
include { GUNZIP_BAM }  from '../modules/gunzip.nf'
include { PBMM2_ALIGN } from '../modules/pbmm2_align.nf'
include { INDEX_BAM }   from '../modules/index_bam.nf'
include { IPDSUMMARY }  from '../modules/ipdsummary.nf'

workflow PREPARE {

    // -------- 1. Parse samplesheet --------
    ch_samples = Channel.fromPath(params.samplesheet, checkIfExists: true)
        .splitCsv(header: true)
        .map { r ->
            def ref = r.reference?.trim()
            tuple(r.sample_id, file(r.reads, checkIfExists: true), ref ? file(ref) : null)
        }

    // -------- 2. Reads: decompress if .bam.gz --------
    ch_reads_raw = ch_samples.map { sid, reads, _ref -> tuple(sid, reads) }

    ch_reads_raw
        .branch {
            gz:  it[1].name.endsWith('.gz')
            bam: true
        }
        .set { ch_branched_reads }

    ch_decompressed = GUNZIP_BAM(ch_branched_reads.gz).mix(ch_branched_reads.bam)

    // -------- 3. Bystrandify if platform == 'ccs' --------
    ch_ready = (params.platform == 'ccs') ? BYSTRANDIFY(ch_decompressed) : ch_decompressed

    // -------- 4. Reference: assemble if missing --------
    ch_samples
        .branch {
            need_asm: it[2] == null
            have_ref: true
        }
        .set { ch_branched_refs }

    if (params.assemble) {
        ch_assembled = HIFIASM(ch_branched_refs.need_asm.map { sid, reads, _ -> tuple(sid, reads) })
    } else {
        ch_branched_refs.need_asm
            .map { sid, _, _ -> error "Sample '${sid}' has no reference column and params.assemble=false. Either provide a reference in the samplesheet or set --assemble true." }
        ch_assembled = Channel.empty()
    }

    ch_ref = ch_assembled.mix(ch_branched_refs.have_ref.map { sid, _, ref -> tuple(sid, ref) })

    // -------- 5. Align --------
    def preset = params.pbmm2_preset ?: (params.platform == 'subread' ? 'SUBREAD' : 'CCS')

    ch_align_in = ch_ready.join(ch_ref, by: 0, failOnMismatch: true)
                          .map { sid, reads, ref -> tuple(sid, reads, ref, preset) }

    ch_aligned = PBMM2_ALIGN(ch_align_in)

    // -------- 6. Index --------
    ch_indexed = INDEX_BAM(ch_aligned)

    // -------- 7. ipdSummary --------
    ch_ipd = IPDSUMMARY(ch_indexed)

    // -------- 8. Emit manifest (gather) --------
    ch_manifest = ch_ipd
        .map { sid, gff, _csv, bam, _ref ->
            // absolute, resolved paths so extract+merge can read them directly
            "${sid},${bam.toRealPath()},,${gff.toRealPath()}"
        }
        .collectFile(
            name:     params.manifest_name ?: 'manifest_gff.csv',
            seed:     'sample_id,bam_path,motifs,gff',
            newLine:  true,
            sort:     true,
            storeDir: params.outdir
        )

    emit:
    samples  = ch_ipd
    manifest = ch_manifest
}
