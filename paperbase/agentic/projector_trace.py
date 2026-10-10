"""Projector structural observability only. Never stores source or output text."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import re

from .grounding import verified_support_ids
from .projection_semantics import build_projection_semantic_request, projection_semantic_fingerprint
from paperbase.prompts.evidence_projector import EVIDENCE_PROJECTOR_PROMPT_VERSION

SUMMARY_VERSION = 'projector-structure-v3-bounded-span-envelope'
_SHAPES = {'fact','mechanism','comparison','enumeration'}
_OWNERS = {'this_paper','prior_work','unknown'}
_STATUSES = {'qualified','irrelevant'}
_REJECTION_CODES = {'missing_source_anchor','not_proposed','source_anchor_not_found',
    'source_span_missing','source_span_not_found','source_spans_duplicate',
    'source_spans_out_of_order','source_spans_non_contiguous','source_spans_too_broad',
    'fact_ownership_unknown','requirement_id_mismatch','no_attributed_entity',
    'cache_integrity_revalidation','empty_canonical_name','not_present_in_evidence',
    'invalid_attribution','duplicate'}


def _hash(text):
    return sha256(text.encode('utf-8')).hexdigest()


def _id(value, known):
    # Known input IDs are trusted. Unknown IDs retain only a bounded Requirement
    # locator such as R99, never arbitrary provider strings or names.
    return value if isinstance(value,str) and (value in known or re.fullmatch(r'R[0-9]{1,8}',value)) else None


def _ids(value, known):
    return [_id(v,known) for v in value[:64] if _id(v,known) is not None] if isinstance(value,list) else None


def _span_ids(value):
    return [v for v in value[:16] if isinstance(v,str) and re.fullmatch(r'S[1-9][0-9]{0,7}',v)] if isinstance(value,list) else None


def _enum(value, allowed):
    return value if isinstance(value,str) and value in allowed else None


def _projection_summary(item, known):
    if not isinstance(item,dict):return {'object':False}
    supported=item.get('supported_requirement_ids');candidates=item.get('support_candidates')
    summary=dict(object=True,status=_enum(item.get('status'),_STATUSES),
        status_field_present='status' in item,projection_type=_enum(item.get('projection_type'),_SHAPES),
        projection_type_field_present='projection_type' in item,
        fact_type_present=isinstance(item.get('fact_type'),str) and bool(item['fact_type'].strip()),
        value_present=isinstance(item.get('value'),str) and bool(item['value'].strip()),
        mechanism_present=isinstance(item.get('mechanism'),str) and bool(item['mechanism'].strip()),
        ownership=_enum(item.get('ownership'),_OWNERS),ownership_field_present='ownership' in item,
        supported_requirement_ids_field_present='supported_requirement_ids' in item,
        supported_requirement_ids=_ids(supported,known),
        supported_requirement_id_count=len(supported) if isinstance(supported,list) else None,
        support_candidates_field_present='support_candidates' in item,
        support_candidate_count=len(candidates) if isinstance(candidates,list) else None,
        support_candidate_requirement_ids=None,support_candidates=None)
    if isinstance(supported,list):
        summary['unrecordable_supported_id_count']=sum(_id(v,known) is None for v in supported)
        summary['supported_ids_truncated']=len(supported)>64
    if isinstance(candidates,list):
        summary['support_candidate_requirement_ids']=[_id(c.get('requirement_id'),known)
            for c in candidates[:64] if isinstance(c,dict)]
        summary['support_candidates_truncated']=len(candidates)>64
        summary['support_candidates']=[dict(requirement_id=_id(c.get('requirement_id'),known),
            requirement_id_field_present='requirement_id' in c,
            source_span_ids_field_present='source_span_ids' in c,
            source_span_ids=_span_ids(c.get('source_span_ids')),
            source_span_id_count=len(c['source_span_ids']) if isinstance(c.get('source_span_ids'),list) else None,
            support_scope_present=isinstance(c.get('support_scope'),str) and bool(c['support_scope'].strip()),
            unresolved_aspect_count=len(c['unresolved_aspects']) if isinstance(c.get('unresolved_aspects'),list) else None)
            if isinstance(c,dict) else {'object':False} for c in candidates[:64]]
    for field in ('steps','inputs','outputs','comparison_dimensions','entities'):
        summary[field+'_count']=len(item[field]) if isinstance(item.get(field),list) else None
    return summary


def pre_parse_summary(*,content,request,records,projection_type,schema_name,schema,
                      schema_version,call_mode,attempt,fallback_reason=None):
    """json.loads-only inspection before ANY production typed parser/converter.

    Failure to observe never changes the existing execution/recovery policy.
    """
    try:
        known={r.requirement_id for r in request.plan.requirements}
        identities={('source:'+r.fingerprint if call_mode=='batch' else r.evidence_id):r.evidence_id for r in records}
        semantic=build_projection_semantic_request(question=request.question,plan=request.plan,
            projection_type=projection_type,requirement_ids=request.projection_requirement_ids)
        summary=dict(summary_version=SUMMARY_VERSION,provider_call=True,source='completion_content',
            content_sha256=_hash(content),content_chars=len(content),json_parseable=False,
            projection_type=projection_type.value,projection_requirement_ids=list(request.projection_requirement_ids),
            prompt_version=EVIDENCE_PROJECTOR_PROMPT_VERSION,schema_name=schema_name,schema_version=schema_version,
            schema_sha256=_hash(json.dumps(schema,ensure_ascii=False,sort_keys=True,separators=(',',':'))),
            semantic_request_fingerprint=projection_semantic_fingerprint(semantic),call_mode=call_mode,
            attempt=attempt,repair=attempt>1,fallback_reason=fallback_reason,item_count=None,items=None)
        try:payload=json.loads(content)
        except (ValueError,TypeError):return summary
        summary['json_parseable']=True
        if call_mode=='batch':
            raw_items=payload.get('items') if isinstance(payload,dict) else None
        else:
            parts=payload.get('projections') if isinstance(payload,dict) else None
            if not isinstance(parts,list):raw_items=None
            else:
                # Single wire repeats runtime ID per projection. Preserve raw
                # ordinals; group only identical known input runtime IDs.
                grouped={}
                for ordinal,part in enumerate(parts):
                    raw_id=part.get('evidence_id') if isinstance(part,dict) else None
                    key=raw_id if isinstance(raw_id,str) and raw_id in identities else ('unknown',ordinal)
                    entry=grouped.setdefault(key,{'evidence_id':raw_id,'projections':[],'projection_ordinals':[]})
                    entry['projections'].append(part);entry['projection_ordinals'].append(ordinal)
                raw_items=list(grouped.values())
        if not isinstance(raw_items,list):return summary
        summary['item_count']=len(raw_items);summary['items_truncated']=len(raw_items)>32
        summary['items']=[]
        for ordinal,item in enumerate(raw_items[:32]):
            if not isinstance(item,dict):summary['items'].append({'item_ordinal':ordinal,'object':False});continue
            raw_id=item.get('evidence_id');known_id=isinstance(raw_id,str) and raw_id in identities
            parts=item.get('projections')
            entry=dict(item_ordinal=ordinal,object=True,known_input_id=known_id,
                evidence_id=raw_id if known_id else None,runtime_evidence_id=identities[raw_id] if known_id else None,
                evidence_id_field_present='evidence_id' in item,
                projection_count=len(parts) if isinstance(parts,list) else None,
                projections=[_projection_summary(p,known) for p in parts[:8]] if isinstance(parts,list) else None,
                projections_truncated=len(parts)>8 if isinstance(parts,list) else False)
            if 'projection_ordinals' in item:entry['projection_ordinals']=item['projection_ordinals'][:8]
            summary['items'].append(entry)
        return summary
    except Exception:
        return None


def _rejection_code(reason):
    if reason.startswith('support_rejected:'):code=reason.rsplit(':',1)[-1]
    elif reason.startswith('entity_rejected:'):code=reason.partition('reason=')[2].split(':',1)[0]
    elif reason.startswith('support_rejection_detail:'):return 'support_rejection_detail'
    else:return 'unrecognized_rejection'
    return code if code in _REJECTION_CODES else 'unrecognized_rejection'


def post_conversion_summary(*,analyses,records,request,conversion_succeeded):
    """Uses the existing verified_support_ids authority; never a second judge."""
    try:
        by_id={r.evidence_id:r for r in records};items=[]
        for analysis in analyses:
            record=by_id[analysis.evidence_id];projections=analysis.projections
            proposed_available=all(hasattr(p,'proposed_requirement_ids') for p in projections)
            items.append(dict(runtime_evidence_id=analysis.evidence_id,analysis_status=analysis.status.value,
                evaluated_requirement_ids=list(analysis.evaluated_requirement_ids),projection_count=len(projections),
                projection_types=[p.projection_type.value for p in projections],
                proposed_requirement_ids=list(dict.fromkeys(rid for p in projections for rid in p.proposed_requirement_ids)) if proposed_available else None,
                supported_requirement_ids=list(dict.fromkeys(rid for p in projections for rid in p.supported_requirement_ids)),
                support_anchor_count=sum(len(p.support_anchors) for p in projections),
                support_anchor_requirement_ids=[a.requirement_id for p in projections for a in p.support_anchors],
                projection_rejection_count=len(analysis.projection_rejections),
                projection_rejection_codes=list(dict.fromkeys(_rejection_code(r) for r in analysis.projection_rejections)),
                verified_support_ids=list(verified_support_ids(replace(record,assessment=analysis),request.plan.requirements))))
        return dict(summary_version=SUMMARY_VERSION,conversion_succeeded=conversion_succeeded,
            converted_item_count=len(items),input_item_count=len(records),items=items)
    except Exception:
        return None
