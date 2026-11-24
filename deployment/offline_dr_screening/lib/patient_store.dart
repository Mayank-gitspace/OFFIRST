import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:path_provider/path_provider.dart';

String recordId(String prefix) {
  final random = Random.secure();
  final suffix = List.generate(
      16, (_) => random.nextInt(256).toRadixString(16).padLeft(2, '0')).join();
  return '$prefix-$suffix';
}

enum Eye {
  right('OD', 'Right eye'),
  left('OS', 'Left eye');

  const Eye(this.code, this.label);
  final String code;
  final String label;
  static Eye fromCode(String code) =>
      values.singleWhere((eye) => eye.code == code);
}

class PatientRecord {
  PatientRecord(
      {required this.id,
      required this.name,
      required this.createdAt,
      this.age,
      this.notes,
      List<VisitRecord>? visits})
      : visits = visits ?? [];
  final String id;
  final String name;
  final DateTime createdAt;
  final int? age;
  final String? notes;
  final List<VisitRecord> visits;

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'createdAt': createdAt.toIso8601String(),
        'age': age,
        'notes': notes,
        'visits': visits.map((v) => v.toJson()).toList(),
      };

  factory PatientRecord.fromJson(Map<String, dynamic> json) => PatientRecord(
        id: json['id'],
        name: json['name'],
        createdAt: DateTime.parse(json['createdAt']),
        age: json['age'],
        notes: json['notes'],
        visits: (json['visits'] as List)
            .indexed
            .map((entry) => VisitRecord.fromJson(entry.$2,
                legacyId: '${json['id']}-legacy-${entry.$1}'))
            .toList(),
      );
}

class ImageRecord {
  const ImageRecord(
      {required this.id,
      required this.eye,
      required this.capturedAt,
      required this.path,
      required this.probability,
      required this.threshold,
      required this.modelVersion});
  final String id;
  final Eye eye;
  final DateTime capturedAt;
  final String path;
  final double probability;
  final double threshold;
  final String modelVersion;
  bool get flagged => probability >= threshold;

  Map<String, dynamic> toJson() => {
        'id': id,
        'eye': eye.code,
        'capturedAt': capturedAt.toIso8601String(),
        'path': path,
        'probability': probability,
        'threshold': threshold,
        'modelVersion': modelVersion,
      };

  factory ImageRecord.fromJson(Map<String, dynamic> json) => ImageRecord(
        id: json['id'],
        eye: Eye.fromCode(json['eye']),
        capturedAt: DateTime.parse(json['capturedAt']),
        path: json['path'],
        probability: (json['probability'] as num).toDouble(),
        threshold: (json['threshold'] as num).toDouble(),
        modelVersion: json['modelVersion'],
      );
}

class VisitRecord {
  VisitRecord(
      {required this.id, required this.date, required List<ImageRecord> images})
      : images = List.unmodifiable(images);
  final String id;
  final DateTime date;
  final List<ImageRecord> images;

  Iterable<ImageRecord> forEye(Eye eye) =>
      images.where((image) => image.eye == eye);

  Map<String, dynamic> toJson() => {
        'id': id,
        'date': date.toIso8601String(),
        'images': images.map((image) => image.toJson()).toList(),
      };

  factory VisitRecord.fromJson(Map<String, dynamic> json,
      {required String legacyId}) {
    final date = DateTime.parse(json['date']);
    if (!json.containsKey('images')) {
      return VisitRecord(id: legacyId, date: date, images: [
        ImageRecord(
            id: '$legacyId-image',
            eye: Eye.fromCode(json['eye']),
            capturedAt: date,
            path: json['imagePath'],
            probability: (json['probability'] as num).toDouble(),
            threshold: 0.48,
            modelVersion: 'offirst-dr-onnx-b8fc1019'),
      ]);
    }
    return VisitRecord(
        id: json['id'],
        date: date,
        images: (json['images'] as List)
            .map((image) => ImageRecord.fromJson(image))
            .toList());
  }
}

class PatientStore {
  PatientStore({Future<Directory> Function()? directory})
      : _directory = directory ?? getApplicationSupportDirectory;
  final Future<Directory> Function() _directory;
  Future<Directory> get directory => _directory();
  static Future<void> _pending = Future.value();

  Future<File> get _file async =>
      File('${(await directory).path}/offirst_patients.json');

  Future<List<PatientRecord>> load() async {
    final file = await _file;
    if (!await file.exists()) return [];
    final entries = jsonDecode(await file.readAsString()) as List;
    return entries.map((entry) => PatientRecord.fromJson(entry)).toList();
  }

  Future<void> addPatient(PatientRecord patient) => _update((patients) async {
        patients.insert(0, patient);
      });

  /// Copy only at finalization. Picker paths may disappear between launches.
  Future<void> addVisit(String patientId, VisitRecord visit) =>
      _update((patients) async {
        final patient =
            patients.singleWhere((record) => record.id == patientId);
        if (patient.visits.any((record) => record.id == visit.id)) return;
        final root = await directory;
        final saved = <ImageRecord>[];
        for (final image in visit.images) {
          final relative = 'visits/${visit.id}/${image.eye.code}/${image.id}';
          final target = File('${root.path}/$relative');
          await target.parent.create(recursive: true);
          await File(image.path).copy(target.path);
          saved.add(ImageRecord(
              id: image.id,
              eye: image.eye,
              capturedAt: image.capturedAt,
              path: relative,
              probability: image.probability,
              threshold: image.threshold,
              modelVersion: image.modelVersion));
        }
        patient.visits.insert(
            0, VisitRecord(id: visit.id, date: visit.date, images: saved));
      });

  Future<File> original(ImageRecord image) async {
    if (!image.path.startsWith('visits/')) return File(image.path);
    return File('${(await directory).path}/${image.path}');
  }

  Future<void> _update(Future<void> Function(List<PatientRecord>) change) {
    final operation = _pending.then((_) async {
      final patients = await load();
      await change(patients);
      final file = await _file;
      await file.parent.create(recursive: true);
      final staged = File('${file.path}.staged');
      await staged.writeAsString(
          jsonEncode(patients.map((p) => p.toJson()).toList()),
          flush: true);
      await staged.rename(file.path);
    });
    _pending =
        operation.then<void>((_) {}, onError: (Object _, StackTrace __) {});
    return operation;
  }
}
