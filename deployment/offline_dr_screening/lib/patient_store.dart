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
      List<String>? expertIds,
      List<VisitRecord>? visits})
      : visits = visits ?? [],
        expertIds = expertIds ?? [];
  final String id;
  final String name;
  final DateTime createdAt;
  final int? age;
  final String? notes;
  final List<String> expertIds;
  final List<VisitRecord> visits;

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'createdAt': createdAt.toIso8601String(),
        'age': age,
        'notes': notes,
        'expertIds': expertIds,
        'visits': visits.map((v) => v.toJson()).toList(),
      };

  factory PatientRecord.fromJson(Map<String, dynamic> json) => PatientRecord(
        id: json['id'],
        name: json['name'],
        createdAt: DateTime.parse(json['createdAt']),
        age: json['age'],
        notes: json['notes'],
        expertIds: (json['expertIds'] as List?)?.cast<String>() ?? const [],
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

class ExpertRecord {
  const ExpertRecord({
    required this.id,
    required this.name,
    required this.updatedAt,
    this.title,
    this.specialty,
    this.clinic,
    this.phone,
    this.email,
    this.city,
    this.address,
    this.schedule,
    this.notes,
  });
  final String id;
  final String name;
  final DateTime updatedAt;
  final String? title;
  final String? specialty;
  final String? clinic;
  final String? phone;
  final String? email;
  final String? city;
  final String? address;
  final String? schedule;
  final String? notes;

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'updatedAt': updatedAt.toIso8601String(),
        'title': title,
        'specialty': specialty,
        'clinic': clinic,
        'phone': phone,
        'email': email,
        'city': city,
        'address': address,
        'schedule': schedule,
        'notes': notes,
      };

  factory ExpertRecord.fromJson(Map<String, dynamic> json) => ExpertRecord(
        id: json['id'],
        name: json['name'],
        updatedAt: DateTime.parse(json['updatedAt']),
        title: json['title'],
        specialty: json['specialty'],
        clinic: json['clinic'],
        phone: json['phone'],
        email: json['email'],
        city: json['city'],
        address: json['address'],
        schedule: json['schedule'],
        notes: json['notes'],
      );
}

class PatientStore {
  PatientStore({Future<Directory> Function()? directory})
      : _directory = directory ?? getApplicationSupportDirectory;
  final Future<Directory> Function() _directory;
  Future<Directory> get directory => _directory();
  static Future<void> _pending = Future.value();
  static const _pendingVisitMarker = '.offirst-pending-visit';
  static final _safeId = RegExp(r'^[A-Za-z0-9-]{1,120}$');
  bool _reconciled = false;

  Future<File> get _file async =>
      File('${(await directory).path}/offirst_patients.json');
  Future<File> get _expertsFile async =>
      File('${(await directory).path}/offirst_experts.json');

  Future<List<PatientRecord>> load() => _serialize(_loadRecords);

  Future<List<PatientRecord>> _loadRecords() async {
    final file = await _file;
    if (!await file.exists()) return [];
    final entries = jsonDecode(await file.readAsString()) as List;
    final patients =
        entries.map((entry) => PatientRecord.fromJson(entry)).toList();
    // Only reconcile after successfully reading every record. A missing or
    // corrupt database is never evidence that a visit may be removed.
    if (!_reconciled) _reconciled = await _reconcile(file.parent, patients);
    return patients;
  }

  Future<void> addPatient(PatientRecord patient) => _serialize(() async {
        final patients = await _loadRecords();
        if (patients.any((record) => record.id == patient.id)) {
          throw StateError('Patient identifier already exists.');
        }
        patients.insert(0, patient);
        await _writeMetadata(patients);
      });

  Future<List<ExpertRecord>> loadExperts() => _serialize(() async {
        final file = await _expertsFile;
        if (!await file.exists()) return <ExpertRecord>[];
        final data = jsonDecode(await file.readAsString()) as List;
        return data
            .map(
                (entry) => ExpertRecord.fromJson(entry as Map<String, dynamic>))
            .toList();
      });

  Future<void> addExpert(ExpertRecord expert) => _serialize(() async {
        final experts = await _loadExperts();
        if (experts.any((record) => record.id == expert.id)) {
          throw StateError('Expert identifier already exists.');
        }
        experts.insert(0, expert);
        await _writeExperts(experts);
      });

  Future<void> updateExpert(ExpertRecord expert) => _serialize(() async {
        final experts = await _loadExperts();
        final index = experts.indexWhere((record) => record.id == expert.id);
        if (index < 0) throw StateError('Expert record does not exist.');
        experts[index] = expert;
        await _writeExperts(experts);
      });

  Future<void> deleteExpert(String expertId) => _serialize(() async {
        final patients = await _loadRecords();
        var associationsChanged = false;
        for (var i = 0; i < patients.length; i++) {
          final patient = patients[i];
          if (!patient.expertIds.contains(expertId)) continue;
          associationsChanged = true;
          patients[i] = PatientRecord(
            id: patient.id,
            name: patient.name,
            createdAt: patient.createdAt,
            age: patient.age,
            notes: patient.notes,
            expertIds: patient.expertIds.where((id) => id != expertId).toList(),
            visits: patient.visits,
          );
        }
        final experts = await _loadExperts();
        if (!experts.any((expert) => expert.id == expertId)) {
          throw StateError('Expert record does not exist.');
        }
        if (associationsChanged) await _writeMetadata(patients);
        await _writeExperts(
            experts.where((expert) => expert.id != expertId).toList());
      });

  Future<void> setPatientExperts(String patientId, List<String> expertIds) =>
      _serialize(() async {
        if (expertIds.toSet().length != expertIds.length) {
          throw ArgumentError('Duplicate expert association.');
        }
        final patients = await _loadRecords();
        final index = patients.indexWhere((patient) => patient.id == patientId);
        if (index < 0) throw StateError('Patient record does not exist.');
        final known = (await _loadExperts()).map((expert) => expert.id).toSet();
        if (!expertIds.every(known.contains)) {
          throw StateError('Unknown expert in association.');
        }
        final patient = patients[index];
        patients[index] = PatientRecord(
          id: patient.id,
          name: patient.name,
          createdAt: patient.createdAt,
          age: patient.age,
          notes: patient.notes,
          expertIds: List.unmodifiable(expertIds),
          visits: patient.visits,
        );
        await _writeMetadata(patients);
      });

  Future<List<ExpertRecord>> _loadExperts() async {
    final file = await _expertsFile;
    if (!await file.exists()) return [];
    final data = jsonDecode(await file.readAsString()) as List;
    return data
        .map((entry) => ExpertRecord.fromJson(entry as Map<String, dynamic>))
        .toList();
  }

  Future<void> _writeExperts(List<ExpertRecord> experts) async {
    final file = await _expertsFile;
    await file.parent.create(recursive: true);
    final staged = File('${file.path}.staged');
    try {
      await staged.writeAsString(
          jsonEncode(experts.map((expert) => expert.toJson()).toList()),
          flush: true);
      await staged.rename(file.path);
    } catch (_) {
      try {
        if (await staged.exists()) await staged.delete();
      } on FileSystemException {
        // Retain the underlying write failure.
      }
      rethrow;
    }
  }

  Future<void> updatePatient(PatientRecord updated) => _serialize(() async {
        final patients = await _loadRecords();
        final index = patients.indexWhere((record) => record.id == updated.id);
        if (index < 0) throw StateError('Patient record does not exist.');
        final existing = patients[index];
        patients[index] = PatientRecord(
          id: existing.id,
          name: updated.name,
          createdAt: existing.createdAt,
          age: updated.age,
          notes: updated.notes,
          expertIds: existing.expertIds,
          visits: existing.visits,
        );
        await _writeMetadata(patients);
      });

  /// Removes metadata first, then only visit directories referenced by the
  /// deleted patient. A failed file cleanup leaves unreferenced private files,
  /// but cannot damage another patient's records or report a false success.
  Future<void> deletePatient(String patientId) => _serialize(() async {
        final patients = await _loadRecords();
        final index = patients.indexWhere((record) => record.id == patientId);
        if (index < 0) throw StateError('Patient record does not exist.');
        final removed = patients[index];
        final remaining = List<PatientRecord>.of(patients)..removeAt(index);
        await _writeMetadata(remaining);
        final retainedVisitIds = remaining
            .expand((patient) => patient.visits.map((visit) => visit.id))
            .toSet();
        final root = await directory;
        final visitsRoot = Directory('${root.path}/visits');
        for (final visit in removed.visits) {
          if (!_safeId.hasMatch(visit.id) ||
              retainedVisitIds.contains(visit.id)) {
            continue;
          }
          final visitDirectory = Directory('${visitsRoot.path}/${visit.id}');
          if (await FileSystemEntity.type(visitDirectory.path,
                  followLinks: false) ==
              FileSystemEntityType.directory) {
            await visitDirectory.delete(recursive: true);
          }
          final reportFolder = Directory('${root.path}/screening_reports');
          for (final reportName in [
            'OFFIRST-${visit.id}.pdf',
            'OFFIRST-${visit.id}-v2.pdf',
          ]) {
            final report = File('${reportFolder.path}/$reportName');
            if (await FileSystemEntity.type(report.path, followLinks: false) ==
                FileSystemEntityType.file) {
              await report.delete();
            }
          }
        }
      });

  /// Copy only at finalization. Picker paths may disappear between launches.
  Future<void> addVisit(String patientId, VisitRecord visit) =>
      _serialize(() async {
        final patients = await _loadRecords();
        final patient =
            patients.singleWhere((record) => record.id == patientId);
        if (patient.visits.any((record) => record.id == visit.id)) return;
        final imageIds = visit.images.map((image) => image.id).toSet();
        if (!_safeId.hasMatch(visit.id) ||
            imageIds.length != visit.images.length ||
            !imageIds.every(_safeId.hasMatch)) {
          throw ArgumentError('Invalid or duplicate visit/image identifier.');
        }
        final root = await directory;
        final visits = Directory('${root.path}/visits');
        await visits.create(recursive: true);
        final destination = '${visits.path}/${visit.id}';
        if (await FileSystemEntity.type(destination, followLinks: false) !=
            FileSystemEntityType.notFound) {
          throw StateError('Visit image directory already exists.');
        }

        var staged = await visits.createTemp('.staged-');
        try {
          // Move the marker with the images so startup can identify an
          // interrupted transaction on either side of the directory rename.
          await File('${staged.path}/$_pendingVisitMarker')
              .writeAsString(visit.id, flush: true);
          final saved = <ImageRecord>[];
          for (final image in visit.images) {
            final imagePath = '${image.eye.code}/${image.id}';
            final target = File('${staged.path}/$imagePath');
            await target.parent.create(recursive: true);
            await File(image.path).copy(target.path);
            saved.add(ImageRecord(
                id: image.id,
                eye: image.eye,
                capturedAt: image.capturedAt,
                path: 'visits/${visit.id}/$imagePath',
                probability: image.probability,
                threshold: image.threshold,
                modelVersion: image.modelVersion));
          }
          staged = await staged.rename(destination);
          patient.visits.insert(
              0, VisitRecord(id: visit.id, date: visit.date, images: saved));
          await _writeMetadata(patients);
        } catch (_) {
          try {
            await staged.delete(recursive: true);
          } on FileSystemException {
            // Preserve the original error. Leave the marker for a retry.
            _reconciled = false;
          }
          rethrow;
        }
        try {
          await File('${staged.path}/$_pendingVisitMarker').delete();
        } on FileSystemException {
          // Metadata is already committed. Recovery will preserve the images.
          _reconciled = false;
        }
      });

  Future<bool> _reconcile(Directory root, List<PatientRecord> patients) async {
    final visits = Directory('${root.path}/visits');
    String platformCase(String path) =>
        Platform.isWindows ? path.toLowerCase() : path;
    final prefix = platformCase(visits.absolute.uri.normalizePath().path);
    final referenced = <String>{};
    for (final patient in patients) {
      for (final visit in patient.visits) {
        referenced.add(platformCase(visit.id));
        for (final image in visit.images) {
          final file = image.path.startsWith('visits/')
              ? File('${root.path}/${image.path}')
              : File(image.path);
          final path = platformCase(file.absolute.uri.normalizePath().path);
          if (path.startsWith(prefix)) {
            referenced.add(Uri.decodeComponent(
                path.substring(prefix.length).split('/').first));
          }
        }
      }
    }

    try {
      if (await FileSystemEntity.type(visits.path, followLinks: false) !=
          FileSystemEntityType.directory) {
        return true;
      }
      var complete = true;
      await for (final entry in visits.list(followLinks: false)) {
        if (entry is! Directory) continue;
        try {
          final name =
              entry.uri.pathSegments.where((part) => part.isNotEmpty).last;
          final marker = File('${entry.path}/$_pendingVisitMarker');
          if (await FileSystemEntity.type(marker.path, followLinks: false) !=
                  FileSystemEntityType.file ||
              await marker.length() > 120) {
            continue;
          }
          final id = await marker.readAsString();
          if (!_safeId.hasMatch(id) ||
              (name != id && !name.startsWith('.staged-'))) {
            continue;
          }
          if (referenced.contains(platformCase(name)) ||
              referenced.contains(platformCase(id))) {
            if (name == id) await marker.delete();
          } else {
            await entry.delete(recursive: true);
          }
        } on FileSystemException {
          complete = false;
        } on FormatException {
          // A marker that cannot be decoded is not proof of ownership.
        }
      }
      return complete;
    } on FileSystemException {
      return false;
    }
  }

  Future<File> original(ImageRecord image) async {
    if (!image.path.startsWith('visits/')) return File(image.path);
    return File('${(await directory).path}/${image.path}');
  }

  Future<void> _writeMetadata(List<PatientRecord> patients) async {
    final file = await _file;
    await file.parent.create(recursive: true);
    final staged = File('${file.path}.staged');
    try {
      await staged.writeAsString(
          jsonEncode(patients.map((p) => p.toJson()).toList()),
          flush: true);
      await staged.rename(file.path);
    } catch (_) {
      try {
        if (await staged.exists()) await staged.delete();
      } on FileSystemException {
        // Keep the failure from the metadata write, not its cleanup.
      }
      rethrow;
    }
  }

  Future<T> _serialize<T>(Future<T> Function() change) {
    final operation = _pending.then((_) => change());
    _pending =
        operation.then<void>((_) {}, onError: (Object _, StackTrace __) {});
    return operation;
  }
}
