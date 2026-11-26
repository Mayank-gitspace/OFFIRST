import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker/image_picker.dart';
import 'package:offirst_dr_screening/patient_store.dart';
import 'package:offirst_dr_screening/screening_page.dart';
import 'package:offirst_dr_screening/screening_probes.dart';
import 'package:offirst_dr_screening/screening_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('severity softmax, colours, and folded weights match the probe file',
      () async {
    final raw = jsonDecode(File(severityAsset).readAsStringSync()) as Map;
    final probes = await ScreeningProbes.load();
    final embedding = _unitVector();
    final grade = probes.severity(embedding);

    expect(grade.labels, gradeLabels);
    expect(
        grade.colors.map((color) =>
            '#${color.toARGB32().toRadixString(16).substring(2).toUpperCase()}'),
        gradeColorHex);
    expect(grade.probabilities.reduce((a, b) => a + b), closeTo(1, 1e-12));
    expect(grade.anyDrRisk, closeTo(1 - grade.probabilities[0], 1e-12));
    expect(grade.index, _argmax(grade.probabilities));
    expect(grade.confidence, grade.probabilities[grade.index]);

    final unfolded = _unfoldedProbabilities(raw, embedding);
    for (var index = 0; index < 4; index++) {
      expect(grade.probabilities[index], closeTo(unfolded[index], 1e-8));
    }
  });

  test('multi-disease bands sort by risk and keep the 4-class DR grade',
      () async {
    final raw = jsonDecode(File(multidiseaseAsset).readAsStringSync()) as Map;
    final probes = await ScreeningProbes.load();
    final embedding = _unitVector();
    final severity = probes.severity(embedding);
    final report = probes.multiDisease(embedding);
    final diseases = report.diseases;

    expect(report.summary.code, 'Disease_Risk');
    expect(report.summary.name, 'Any Disease Risk');
    expect(diseases, hasLength(14));
    expect(diseases.map((disease) => disease.code), isNot(contains('Disease_Risk')));
    expect(
        {report.summary.code, ...diseases.map((disease) => disease.code)},
        diseaseOrder.toSet());
    expect(diseases.singleWhere((disease) => disease.code == 'MH').name,
        'Media Haze');
    expect(
        diseases.singleWhere((disease) => disease.code == 'Hypertension').name,
        'Hypertensive Retinopathy');
    for (var index = 1; index < diseases.length; index++) {
      expect(diseases[index - 1].risk,
          greaterThanOrEqualTo(diseases[index].risk));
    }
    final dr = diseases.singleWhere((disease) => disease.code == 'DR');
    expect(dr.usesSeverityHead, isTrue);
    expect(dr.label, severity.label);
    expect(dr.color, severity.color);
    expect(dr.risk, closeTo(severity.anyDrRisk, 1e-12));
    expect(dr.name, 'Diabetic Retinopathy');
    _expectFoldedScores(raw, embedding, report);
  });

  test('colour bands and rare-disease thresholds follow the published cuts',
      () {
    expect(gradeBand(0.10, 0.20).label, 'Low');
    expect(gradeBand(0.30, 0.20).label, 'Borderline');
    expect(gradeBand(0.75, 0.20).label, 'Elevated');
    expect(gradeBand(0.90, 0.20).label, 'High');
    expect(gradeBand(0.70, 0.60).label, 'Borderline');
    expect(gradeBand(0.75, 0.60).label, 'Elevated');
    expect(gradeBand(0.60, 0.60).label, 'Borderline');
    expect(gradeBand(0.88, 0.20).label, 'High');
    expect(gradeBand(0.879, 0.20).label, 'Elevated');
    expect(gradeBand(0.89, 0.90).label, 'Low');
    expect(gradeBand(0.72, 0.5775721073150635).label, 'Borderline');
    expect(gradeBand(0.73, 0.5775721073150635).label, 'Elevated');
    const hypertensionYouden = 0.15153447541171228;
    const hypertensionPrev = 0.8037505256809883;
    expect(
        operatingThreshold(
            'Hypertension', hypertensionYouden, hypertensionPrev),
        hypertensionPrev);
    expect(operatingThreshold('ARMD', 0.34125672827850795, 0.8099401749927814),
        0.35);
    expect(operatingThreshold('ODC', 0.42605588973422226, 0.6644538591327153),
        0.42605588973422226);
    expect(
        gradeBand(0.50, hypertensionPrev).label,
        'Low');
    expect(gradeBand(0.85, hypertensionPrev).label, 'Borderline');
    expect(gradeBand(0.96, hypertensionPrev).label, 'High');
  });

  test('pdf signal uses Youden and the 0.70 / 0.85 cuts', () {
    expect(reportSignal(0.10, 0.20), isNull);
    expect(reportSignal(0.20, 0.25), isNull);
    expect(reportSignal(0.20, 0.15), 'Borderline');
    expect(reportSignal(0.69, 0.15), 'Borderline');
    expect(reportSignal(0.70, 0.15), 'Moderate');
    expect(reportSignal(0.849, 0.15), 'Moderate');
    expect(reportSignal(0.85, 0.15), 'High / Marked');
    expect(reportSignal(0.91, 0.15), 'High / Marked');
    expect(reportSignal(0.84, 0.90), isNull);
    expect(reportSignal(0.90, 0.90), 'High / Marked');
  });

  test('pdf findings follow raw Youden and omit DR and disease risk', () async {
    final raw = jsonDecode(File(multidiseaseAsset).readAsStringSync()) as Map;
    final probes = await ScreeningProbes.load();
    final embedding = _unitVector();
    final findings = probes.screeningFindings(embedding);
    expect(
        findings.map((finding) => finding.code),
        diseaseOrder.where((code) => code != 'DR' && code != 'Disease_Risk'));
    final heads = raw['heads'] as Map;
    for (final finding in findings) {
      final head = heads[finding.code] as Map;
      final weights = (head['w_folded'] as List).cast<num>();
      var logit = (head['b_folded'] as num).toDouble();
      for (var index = 0; index < 768; index++) {
        logit += weights[index].toDouble() * embedding[index];
      }
      final probability = logit >= 0
          ? 1 / (1 + math.exp(-logit))
          : math.exp(logit) / (1 + math.exp(logit));
      expect(finding.probability, closeTo(probability, 1e-8));
      expect(finding.name, head['full_name']);
      expect(
          finding.signal,
          reportSignal(
              probability, (head['thr_youden_cv'] as num).toDouble()));
    }
  });

  test('image records keep a cached embedding and omit it when absent', () {
    final plain = ImageRecord(
      id: 'IMG-plain',
      eye: Eye.right,
      capturedAt: DateTime.utc(2026),
      path: 'a.jpg',
      probability: 0.2,
      threshold: 0.5,
      modelVersion: 'model',
    );
    expect(plain.toJson().containsKey('embedding'), isFalse);
    final stored = ImageRecord(
      id: 'IMG-z',
      eye: Eye.left,
      capturedAt: DateTime.utc(2026),
      path: 'b.jpg',
      probability: 0.4,
      threshold: 0.5,
      modelVersion: 'model',
      embedding: const [0.25, 0.5],
    );
    expect(ImageRecord.fromJson(stored.toJson()).embedding, [0.25, 0.5]);
  });

  testWidgets('switching screening mode does not run the encoder again',
      (tester) async {
    tester.view.physicalSize = const Size(360, 780);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    var calls = 0;
    final embedding = List<double>.generate(768, (index) => index == 0 ? 1.0 : 0.0);
    final service = ScreeningService.forTesting(
      initializeModel: () async {},
      runPrediction: (_) async {
        calls++;
        return ScreeningResult(0.2, embedding: embedding);
      },
    );
    addTearDown(service.dispose);
    await tester.runAsync(ScreeningProbes.load);
    await tester.pumpWidget(MaterialApp(
      home: Scaffold(
        body: ScreeningPage(
          store: _UnusedStore(),
          service: service,
          onSaved: () {},
          patients: Future.value([
            PatientRecord(
              id: 'OFF-GRADE',
              name: 'Synthetic',
              createdAt: DateTime.utc(2026),
            )
          ]),
          pickImages: (_) async => [XFile('fundus.png')],
        ),
      ),
    ));
    await tester.pumpAndSettle();
    expect(find.text('Screening mode'), findsOneWidget);
    expect(find.text('DR severity'), findsOneWidget);

    final scrollable = find.byType(Scrollable).first;
    await tester.scrollUntilVisible(
        find.byType(DropdownButtonFormField<String>), 120,
        scrollable: scrollable);
    await tester.tap(find.byType(DropdownButtonFormField<String>));
    await tester.pumpAndSettle();
    await tester.tap(find.textContaining('OFF-GRADE').last);
    await tester.pumpAndSettle();
    final add = find.widgetWithText(FilledButton, 'Add fundus images');
    await tester.scrollUntilVisible(add, 120, scrollable: scrollable);
    await tester.tap(add);
    await tester.pumpAndSettle();
    expect(calls, 1);

    await tester.scrollUntilVisible(find.textContaining('Any DR risk'), 160,
        scrollable: scrollable);
    expect(find.text('Healthy'), findsWidgets);
    expect(find.text('Mild'), findsWidgets);
    expect(find.text('Moderate'), findsWidgets);
    expect(find.text('Severe'), findsWidgets);
    expect(find.textContaining('Any DR risk'), findsOneWidget);
    expect(find.textContaining('Confidence'), findsOneWidget);
    expect(find.text('Media Haze'), findsNothing);
    expect(find.text('Any Disease Risk'), findsNothing);

    await tester.scrollUntilVisible(find.text('Multi-disease'), -200,
        scrollable: scrollable);
    await tester.tap(find.text('Multi-disease'));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('Any Disease Risk'), 160,
        scrollable: scrollable);
    expect(find.text('Any Disease Risk'), findsOneWidget);
    await tester.scrollUntilVisible(find.text('Media Haze'), 160,
        scrollable: scrollable);
    expect(find.text('Media Haze'), findsOneWidget);
    expect(find.text('Diabetic Retinopathy'), findsOneWidget);
    expect(find.text('Hypertensive Retinopathy'), findsOneWidget);
    expect(find.textContaining('same 4-class grade'), findsOneWidget);
    expect(calls, 1);

    await tester.scrollUntilVisible(find.text('DR severity'), -200,
        scrollable: scrollable);
    await tester.tap(find.text('DR severity'));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.textContaining('Any DR risk'), 160,
        scrollable: scrollable);
    expect(find.text('Media Haze'), findsNothing);
    expect(find.text('Any Disease Risk'), findsNothing);
    expect(calls, 1);
  });
}

List<double> _unitVector() {
  final values = List<double>.generate(768, (index) {
    return ((index * 17) % 100) / 50 - 1;
  });
  var sum = 0.0;
  for (final value in values) {
    sum += value * value;
  }
  final norm = math.sqrt(sum);
  return [for (final value in values) value / norm];
}

int _argmax(List<double> values) {
  var chosen = 0;
  for (var index = 1; index < values.length; index++) {
    if (values[index] > values[chosen]) chosen = index;
  }
  return chosen;
}

void _expectFoldedScores(
    Map raw, List<double> embedding, MultiDiseaseReport report) {
  final heads = raw['heads'] as Map;
  final byCode = {for (final grade in report.diseases) grade.code: grade};
  byCode[report.summary.code] = report.summary;
  for (final code in diseaseOrder) {
    final head = heads[code] as Map;
    final weights = (head['w_folded'] as List).cast<num>();
    final bias = (head['b_folded'] as num).toDouble();
    final mean = (head['scaler_mean'] as List).cast<num>();
    final scale = (head['scaler_scale'] as List).cast<num>();
    final coefficient = (head['coef'] as List).cast<num>();
    final intercept = (head['intercept'] as num).toDouble();
    var folded = bias;
    var unfolded = intercept;
    for (var index = 0; index < 768; index++) {
      folded += weights[index].toDouble() * embedding[index];
      unfolded += coefficient[index].toDouble() *
          ((embedding[index] - mean[index].toDouble()) /
              scale[index].toDouble());
    }
    expect((folded - unfolded).abs(), lessThan(1e-8));
    final probability = folded >= 0
        ? 1 / (1 + math.exp(-folded))
        : math.exp(folded) / (1 + math.exp(folded));
    final grade = byCode[code]!;
    if (code == 'DR') {
      expect(grade.usesSeverityHead, isTrue);
    } else {
      expect(grade.risk, closeTo(probability, 1e-8));
      expect(
          grade.label,
          gradeBand(
                  probability,
                  operatingThreshold(
                      code,
                      (head['thr_youden_cv'] as num).toDouble(),
                      (head['thr_prev_cv'] as num).toDouble()))
              .label);
    }
  }
}

List<double> _unfoldedProbabilities(Map raw, List<double> embedding) {
  final mean = (raw['scaler_mean'] as List).cast<num>();
  final scale = (raw['scaler_scale'] as List).cast<num>();
  final coefficient = raw['lr_coef'] as List;
  final intercept = (raw['lr_intercept'] as List).cast<num>();
  final logits = <double>[];
  for (var row = 0; row < 4; row++) {
    final weights = (coefficient[row] as List).cast<num>();
    var logit = intercept[row].toDouble();
    for (var index = 0; index < 768; index++) {
      logit += weights[index].toDouble() *
          ((embedding[index] - mean[index].toDouble()) /
              scale[index].toDouble());
    }
    logits.add(logit);
  }
  var peak = logits.reduce(math.max);
  final shifted = [for (final logit in logits) math.exp(logit - peak)];
  final sum = shifted.reduce((a, b) => a + b);
  return [for (final value in shifted) value / sum];
}

class _UnusedStore extends PatientStore {}
