"""Inspect pickle opcodes without unpickling, importing classes, or reading large payloads.

Reports storage evidence, not validated annotation coverage or manual prompt provenance.
Standard library only. Source file is opened read-only; a new JSON report is written.
"""
import argparse
import json
import pickletools
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OPS = {ord(op.code): op for op in pickletools.opcodes}
LENGTHS = {'BINBYTES':(4,False), 'SHORT_BINBYTES':(1,False),
           'BINBYTES8':(8,False), 'BYTEARRAY8':(8,False),
           'BINSTRING':(4,True), 'SHORT_BINSTRING':(1,False),
           'BINUNICODE':(4,False), 'SHORT_BINUNICODE':(1,False),
           'BINUNICODE8':(8,False)}
FIELDS = {'session_name','view_mode','mode','curr_img_idx','current_img_width',
          'current_img_height','block_size','current_block','num_blocks','video_name',
          'media_path','max_frame','sam_handler_curr_img_idx','sam_handler_current_stage',
          'media_files','masks','combined_masks','tracking_results','label_handler',
          'sam_handler_labels','sam_handler_object_id_to_label_name',
          'sam_handler_propagation_blocks','idx_to_path'}


class BoundedReader:
    def __init__(self, stream): self.stream=stream
    def read(self, size=-1):
        if not 0 <= size <= 2*1024*1024:
            raise ValueError('Unexpected large opcode argument; stopped without allocating it')
        return self.stream.read(size)
    def readline(self):
        value=self.stream.readline(4097)
        if len(value)>4096: raise ValueError('Oversized line argument')
        return value


def inspect(path):
    before=path.stat(); counts=Counter(); samples=[]; seen=set(); fields=[]
    pending=None; stops=[]; skipped=0; payload_examples=[]; position=0
    result=dict(status='in_progress', file_name=path.name, bytes=before.st_size,
        mtime_ns=before.st_mtime_ns, method='Sequential opcode inspection with seek over byte payloads; no unpickling',
        limitations=['No pickle objects are reconstructed and no referenced code is executed.',
            'Field observations are nearby opcode evidence, not resolved dictionary values.',
            'Mask coverage, manual annotations and identity mapping remain unverified.',
            'Unchanged size/mtime and STOP opcodes do not prove transfer integrity or semantic validity.'])
    with path.open('rb') as stream:
        reader=BoundedReader(stream)
        try:
            while True:
                position=stream.tell(); code=stream.read(1)
                if not code: break
                op=OPS.get(code[0])
                if op is None: raise ValueError(f'Unknown opcode byte {code.hex()}')
                counts[op.name]+=1; arg=None
                if op.name in LENGTHS:
                    width,signed=LENGTHS[op.name]; header=stream.read(width)
                    if len(header)!=width: raise ValueError('Truncated payload length')
                    length=int.from_bytes(header,'little',signed=signed)
                    if length<0 or length>before.st_size-stream.tell():
                        raise ValueError('Payload extends beyond current file size')
                    if 'UNICODE' in op.name and length<=4096:
                        arg=stream.read(length).decode('utf-8','surrogatepass')
                    else:
                        stream.seek(length,1); skipped+=length
                        if len(payload_examples)<12: payload_examples.append(dict(offset=position,opcode=op.name,bytes=length))
                elif op.arg is not None:
                    arg=op.arg.reader(reader)
                if pending and op.name not in {'MEMOIZE','BINPUT','LONG_BINPUT','PUT','FRAME'}:
                    if len(fields)<300:
                        fields.append(dict(field=pending, next_opcode=op.name, offset=position,
                            value=arg if isinstance(arg,(int,float,bool)) else (arg[:250] if isinstance(arg,str) else None)))
                    pending=None
                if isinstance(arg,str):
                    if arg in FIELDS: pending=arg
                    # Image path lists dominate sessions; retain metadata-like strings only.
                    if not arg.lower().endswith(('.png','.jpg','.jpeg')) and len(arg)<=250 and arg not in seen and len(samples)<300:
                        seen.add(arg); samples.append(dict(offset=position,opcode=op.name,text=arg))
                if op.name=='STOP':
                    stops.append(position)
                    pending=None
            result['status']='opcode_scan_completed' if stops and stops[-1]==before.st_size-1 else 'incomplete_pickle_stream'
        except (ValueError,UnicodeError,OverflowError,EOFError) as exc:
            result.update(status='scan_stopped',error=str(exc),error_offset=position)
        result['bytes_traversed']=stream.tell()
    after=path.stat()
    result.update(opcode_counts=dict(counts), stop_offsets=stops, skipped_payload_bytes=skipped,
        payload_examples=payload_examples, field_observations=fields, text_samples=samples,
        source_unchanged=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns))
    if not result['source_unchanged']: result['status']='source_changed_during_scan'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    args=p.parse_args()
    report=inspect(args.session)
    folder=Path(__file__).resolve().parent/'results'
    folder.mkdir(exist_ok=True)
    target=folder/('intake-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    with target.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,ensure_ascii=True)
    print('Status:',report['status'])
    print('Payload bytes skipped:',report['skipped_payload_bytes'])
    print('STOP opcodes:',len(report['stop_offsets']))
    print('Report:',target)
    for entry in report['field_observations']:
        if entry['value'] is not None: print(entry['field'],':',entry['value'])
    return 0 if report['status']=='opcode_scan_completed' else 1


if __name__=='__main__': raise SystemExit(main())
