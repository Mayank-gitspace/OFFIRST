import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/services.dart';

const severityAsset = 'assets/heads/kd768_lp_dr_severity4_aptos.json';
const multidiseaseAsset = 'assets/heads/kd768_lp_multidisease_rfmid_odir.json';

const gradeLabels = ['Healthy', 'Mild', 'Moderate', 'Severe'];
const riskTierLabels = ['Low', 'Borderline', 'Elevated', 'High'];
const gradeColorHex = ['#22C55E', '#EAB308', '#F97316', '#EF4444'];

const diseaseOrder = [
  'Disease_Risk',
  'DR',
  'ARMD',
  'MYA',
  'Glaucoma',
  'Cataract',
  'Hypertension',
  'MH',
  'DN',
  'BRVO',
  'TSLN',
  'LS',
  'ODC',
  'ODP',
  'ODE',
];

enum ScreeningMode { severity, multidisease }

class GradeBand {
  const GradeBand(this.label, this.index);
  final String label;
  final int index;
}

/// Operating cutoff for one multi-disease head.
/// Youden is floored at 0.35. Hypertension uses its prevalence-matched
/// cutoff instead: thr_youden_cv is 0.152 and flags that rare finding too
/// readily, while thr_prev_cv is about 0.804.
double operatingThreshold(String code, double youden, double prevalence) {
  if (code == 'Hypertension') return prevalence;
  return math.max(youden, 0.35);
}

/// PDF multi-disease signal. A probability below [youden] is not flagged.
/// Borderline is below 0.70, Moderate is below 0.85, and High / Marked is
/// at or above 0.85.
String? reportSignal(double probability, double youden) {
  if (probability < youden) return null;
  if (probability >= 0.85) return 'High / Marked';
  if (probability >= 0.70) return 'Moderate';
  return 'Borderline';
}

/// 4-level risk tiers. Low is strictly below [threshold]. Borderline runs up
/// to max(threshold + 0.15, 0.70). Elevated runs up to 0.88. High is at or
/// above 0.88. A probability below [threshold] stays Low.
GradeBand gradeBand(double probability, double threshold) {
  if (probability < threshold) return GradeBand(riskTierLabels[0], 0);
  final mildCut = math.max(threshold + 0.15, 0.70);
  if (probability < mildCut) return GradeBand(riskTierLabels[1], 1);
  if (probability < 0.88) return GradeBand(riskTierLabels[2], 2);
  return GradeBand(riskTierLabels[3], 3);
}

class SeverityGrade {
  const SeverityGrade({
    required this.labels,
    required this.colors,
    required this.probabilities,
    required this.index,
  });

  final List<String> labels;
  final List<Color> colors;
  final List<double> probabilities;
  final int index;

  String get label => labels[index];
  Color get color => colors[index];
  double get confidence => probabilities[index];
  double get anyDrRisk => 1 - probabilities[0];
}

class DiseaseGrade {
  const DiseaseGrade({
    required this.code,
    required this.name,
    required this.risk,
    required this.label,
    required this.color,
    required this.usesSeverityHead,
  });

  final String code;
  final String name;
  final double risk;
  final String label;
  final Color color;
  final bool usesSeverityHead;
}

class MultiDiseaseReport {
  const MultiDiseaseReport({required this.summary, required this.diseases});

  final DiseaseGrade summary;
  final List<DiseaseGrade> diseases;
}

class ScreeningFinding {
  const ScreeningFinding({
    required this.code,
    required this.name,
    required this.probability,
    required this.signal,
  });

  final String code;
  final String name;
  final double probability;

  /// Null when [probability] is below this head's Youden cutoff.
  final String? signal;
}

class _LinearHead {
  const _LinearHead({
    required this.name,
    required this.weights,
    required this.bias,
    required this.threshold,
    required this.youden,
  });

  final String name;
  final List<double> weights;
  final double bias;
  final double threshold;
  final double youden;
}

class ScreeningProbes {
  ScreeningProbes._(this._weights, this._bias, this._colors, this._heads);

  static ScreeningProbes? _cached;
  static Future<ScreeningProbes>? _loading;

  final List<List<double>> _weights;
  final List<double> _bias;
  final List<Color> _colors;
  final Map<String, _LinearHead> _heads;

  static Future<ScreeningProbes> load() async {
    final cached = _cached;
    if (cached != null) return cached;
    final pending = _loading;
    if (pending != null) return pending;
    final future = _read();
    _loading = future;
    try {
      return _cached = await future;
    } catch (_) {
      if (identical(_loading, future)) _loading = null;
      rethrow;
    }
  }

  static Future<ScreeningProbes> _read() async {
    final severity = jsonDecode(await rootBundle.loadString(severityAsset));
    final diseases = jsonDecode(await rootBundle.loadString(multidiseaseAsset));
    return parse(severity, diseases);
  }

  static ScreeningProbes parse(Object? severityJson, Object? diseaseJson) {
    if (severityJson is! Map || diseaseJson is! Map) {
      throw const FormatException('Probe files are not JSON objects.');
    }
    final classes = _stringList(severityJson['classes'], 'classes');
    final colors = _stringList(severityJson['colors'], 'colors');
    if (!_same(classes, gradeLabels) || !_same(colors, gradeColorHex)) {
      throw const FormatException('DR severity classes or colors differ.');
    }
    final weights = _matrix(severityJson['w_folded'], 4, 768, 'w_folded');
    final bias = _vector(severityJson['b_folded'], 4, 'b_folded');
    final headsJson = diseaseJson['heads'];
    if (headsJson is! Map) {
      throw const FormatException('Multi-disease heads are missing.');
    }
    final names = headsJson.keys.map((key) => key.toString()).toSet();
    if (names.length != diseaseOrder.length ||
        !names.containsAll(diseaseOrder)) {
      throw const FormatException('Unexpected multi-disease heads.');
    }
    final listed = diseaseJson['labels'];
    if (listed != null && !_same(_stringList(listed, 'labels'), diseaseOrder)) {
      throw const FormatException('Multi-disease label order differs.');
    }
    final heads = <String, _LinearHead>{};
    for (final code in diseaseOrder) {
      final head = headsJson[code];
      if (head is! Map) {
        throw FormatException('Head $code is missing.');
      }
      final display = head['full_name'];
      if (display is! String || display.trim().isEmpty) {
        throw FormatException('Head $code has no display name.');
      }
      final youden = _finite(head['thr_youden_cv'], '$code.thr_youden_cv');
      final prevalence = _finite(head['thr_prev_cv'], '$code.thr_prev_cv');
      heads[code] = _LinearHead(
        name: display,
        weights: _vector(head['w_folded'], 768, '$code.w_folded'),
        bias: _finite(head['b_folded'], '$code.b_folded'),
        threshold: operatingThreshold(code, youden, prevalence),
        youden: youden,
      );
    }
    return ScreeningProbes._(
      weights,
      bias,
      [for (final hex in gradeColorHex) _color(hex)],
      heads,
    );
  }

  SeverityGrade severity(List<double> embedding) {
    _checkEmbedding(embedding);
    final logits = List<double>.generate(4, (row) {
      final weights = _weights[row];
      var logit = _bias[row];
      for (var index = 0; index < 768; index++) {
        logit += weights[index] * embedding[index];
      }
      return logit;
    }, growable: false);
    final probabilities = _softmax(logits);
    var chosen = 0;
    for (var index = 1; index < probabilities.length; index++) {
      if (probabilities[index] > probabilities[chosen]) chosen = index;
    }
    return SeverityGrade(
      labels: gradeLabels,
      colors: _colors,
      probabilities: probabilities,
      index: chosen,
    );
  }

  MultiDiseaseReport multiDisease(List<double> embedding) {
    final severityGrade = severity(embedding);
    DiseaseGrade? summary;
    final ranked = <({int index, DiseaseGrade grade})>[];
    for (var index = 0; index < diseaseOrder.length; index++) {
      final code = diseaseOrder[index];
      final head = _heads[code]!;
      final probability = _sigmoid(_dot(head.weights, embedding) + head.bias);
      if (code == 'Disease_Risk') {
        final band = gradeBand(probability, head.threshold);
        summary = DiseaseGrade(
          code: code,
          name: head.name,
          risk: probability,
          label: band.label,
          color: _colors[band.index],
          usesSeverityHead: false,
        );
        continue;
      }
      final DiseaseGrade grade;
      if (code == 'DR') {
        grade = DiseaseGrade(
          code: code,
          name: head.name,
          risk: severityGrade.anyDrRisk,
          label: severityGrade.label,
          color: severityGrade.color,
          usesSeverityHead: true,
        );
      } else {
        final band = gradeBand(probability, head.threshold);
        grade = DiseaseGrade(
          code: code,
          name: head.name,
          risk: probability,
          label: band.label,
          color: _colors[band.index],
          usesSeverityHead: false,
        );
      }
      ranked.add((index: index, grade: grade));
    }
    final summaryGrade = summary;
    if (summaryGrade == null) {
      throw StateError('Disease risk summary is missing.');
    }
    ranked.sort((a, b) {
      final byRisk = b.grade.risk.compareTo(a.grade.risk);
      if (byRisk != 0) return byRisk;
      return a.index.compareTo(b.index);
    });
    return MultiDiseaseReport(
      summary: summaryGrade,
      diseases: [for (final entry in ranked) entry.grade],
    );
  }

  /// Non-DR heads for the PDF findings table. Disease risk and the DR head
  /// are omitted; DR severity comes from the 4-class grade.
  List<ScreeningFinding> screeningFindings(List<double> embedding) {
    _checkEmbedding(embedding);
    final findings = <ScreeningFinding>[];
    for (final code in diseaseOrder) {
      if (code == 'Disease_Risk' || code == 'DR') continue;
      final head = _heads[code]!;
      final probability = _sigmoid(_dot(head.weights, embedding) + head.bias);
      findings.add(ScreeningFinding(
        code: code,
        name: head.name,
        probability: probability,
        signal: reportSignal(probability, head.youden),
      ));
    }
    return findings;
  }
}

double _dot(List<double> weights, List<double> embedding) {
  var logit = 0.0;
  for (var index = 0; index < weights.length; index++) {
    logit += weights[index] * embedding[index];
  }
  return logit;
}

void _checkEmbedding(List<double> embedding) {
  if (embedding.length != 768) {
    throw StateError('Embedding rank is not 768.');
  }
  for (final component in embedding) {
    if (!component.isFinite) {
      throw StateError('Embedding contains a non-finite value.');
    }
  }
}

List<double> _softmax(List<double> logits) {
  var peak = logits.first;
  for (final logit in logits) {
    if (logit > peak) peak = logit;
  }
  final shifted = [for (final logit in logits) math.exp(logit - peak)];
  var sum = 0.0;
  for (final value in shifted) {
    sum += value;
  }
  return [for (final value in shifted) value / sum];
}

double _sigmoid(double logit) {
  if (logit >= 0) {
    final z = math.exp(-logit);
    return 1 / (1 + z);
  }
  final z = math.exp(logit);
  return z / (1 + z);
}

Color _color(String hex) =>
    Color(0xFF000000 | int.parse(hex.substring(1), radix: 16));

double _finite(Object? value, String name) {
  if (value is! num) throw FormatException('$name is not a number.');
  final number = value.toDouble();
  if (!number.isFinite) throw FormatException('$name is not finite.');
  return number;
}

List<String> _stringList(Object? value, String name) {
  if (value is! List || value.any((item) => item is! String)) {
    throw FormatException('$name is not a string list.');
  }
  return value.cast<String>();
}

List<double> _vector(Object? value, int length, String name) {
  if (value is! List || value.length != length) {
    throw FormatException('$name must have length $length.');
  }
  return [for (final item in value) _finite(item, name)];
}

List<List<double>> _matrix(Object? value, int rows, int columns, String name) {
  if (value is! List || value.length != rows) {
    throw FormatException('$name must have $rows rows.');
  }
  return [
    for (var row = 0; row < rows; row++)
      _vector(value[row], columns, '$name[$row]')
  ];
}

bool _same(List<String> left, List<String> right) {
  if (left.length != right.length) return false;
  for (var index = 0; index < left.length; index++) {
    if (left[index] != right[index]) return false;
  }
  return true;
}
