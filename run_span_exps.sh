for domain in "$@"
do
  if [ $domain = "geo" ]; then
    python span_based/run_span.py -d geo_iid -w -l
    python span_based/run_span.py -d geo_iid -w
    python span_based/run_span.py -d geo_template -w -l
    python span_based/run_span.py -d geo_template -w
    python span_based/run_span.py -d geo_len -w -l
    python span_based/run_span.py -d geo_len -w
    python span_based/run_span.py -d geo_len_spans
    python span_based/run_span.py -d geo_len_f1 -w -l
    python span_based/run_span.py -d geo_iid_spans
    python span_based/run_span.py -d geo_template_spans
    python span_based/run_span.py -d geo_iid_f1 -w -l
    python span_based/run_span.py -d geo_template_f1 -w -l
  elif [ $domain = "scan" ]; then
    python span_based/run_span.py -d scan_iid -w -l
    python span_based/run_span.py -d scan_iid -w
    python span_based/run_span.py -d scan_right -w -l
    python span_based/run_span.py -d scan_right -w
    python span_based/run_span.py -d scan_around_right -w -l
    python span_based/run_span.py -d scan_around_right -w
    python span_based/run_span.py -d scan_iid_spans
    python span_based/run_span.py -d scan_right_spans
    python span_based/run_span.py -d scan_around_right_spans
    python span_based/run_span.py -d scan_iid_f1 -w -l
    python span_based/run_span.py -d scan_right_f1 -w -l
    python span_based/run_span.py -d scan_around_right_f1 -w -l
  elif [ $domain = "clevr" ]; then
    python span_based/run_span.py -d clevr_iid -w -l
    python span_based/run_span.py -d clevr_iid -w
    python span_based/run_span.py -d clevr_closure -w -l
    python span_based/run_span.py -d clevr_closure -w
    python span_based/run_span.py -d clevr_iid_spans
    python span_based/run_span.py -d clevr_closure_spans
    python span_based/run_span.py -d clevr_iid_f1 -w -l
    python span_based/run_span.py -d clevr_closure_f1 -w -l
  fi
done