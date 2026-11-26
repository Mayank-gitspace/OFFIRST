import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'patient_store.dart';
import 'screening_service.dart';

class ScreeningPage extends StatefulWidget {
  const ScreeningPage(
      {super.key,
      required this.store,
      required this.patients,
      required this.onSaved,
      this.service,
      this.pickImages});
  final ScreeningService? service;
  final Future<List<XFile>> Function(bool camera)? pickImages;
  final PatientStore store;
  final Future<List<PatientRecord>> patients;
  final VoidCallback onSaved;

  @override
  State<ScreeningPage> createState() => _ScreeningPageState();
}

class _ScreeningPageState extends State<ScreeningPage> {
  final _picker = ImagePicker();
  late final _service = widget.service ?? ScreeningService();
  final _failed = <_PendingImage>[];
  final _images = <ImageRecord>[];
  final _unavailableEyes = <Eye>{};
  String _visitId = recordId('VIS');
  String? _patientId;
  Eye _eye = Eye.right;
  String? _message;
  bool _busy = false;

  bool get _hasRetryableFailures => _failed
      .any((image) => image.failure?.stage != ScreeningStage.inputValidation);

  @override
  void initState() {
    super.initState();
    _service.initialize().catchError((Object _) {});
  }

  Future<void> _screen(List<_PendingImage> pending) async {
    for (final image in pending) {
      if (!mounted) break;
      try {
        final result = await _service.predict(File(image.file.path));
        if (!mounted) break;
        setState(() => _images.add(ImageRecord(
              id: recordId('IMG'),
              eye: image.eye,
              capturedAt: image.capturedAt,
              path: image.file.path,
              probability: result.probability,
              threshold: activeScreeningThreshold,
              modelVersion: activeScreeningModelVersion,
            )));
      } catch (error, stack) {
        if (!mounted) break;
        final failure = error is ScreeningFailure
            ? error
            : const ScreeningFailure(ScreeningStage.inference,
                'Inference failed. Retry this image.');
        logScreeningFailure(failure.stage, error, stack);
        image.failure = failure;
        setState(() => _failed.add(image));
      }
    }
  }

  Future<void> _add({required bool camera}) async {
    if (_busy || _patientId == null) return;
    final eye = _eye;
    setState(() {
      _busy = true;
      _message = null;
    });
    try {
      final List<XFile> files;
      if (widget.pickImages != null) {
        files = await widget.pickImages!(camera);
      } else if (camera) {
        final photo = await _picker.pickImage(source: ImageSource.camera);
        files = [if (photo != null) photo];
      } else {
        files = await _picker.pickMultiImage();
      }
      await _screen(files.map((file) => _PendingImage(file, eye)).toList());
    } catch (error, stack) {
      logScreeningFailure(ScreeningStage.imageAccess, error, stack);
      if (mounted) {
        setState(() => _message =
            'Image source could not be opened. Check gallery/camera permission and try again.');
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _retry() async {
    if (_busy || _patientId == null || !_hasRetryableFailures) return;
    final pending = _failed
        .where(
            (image) => image.failure?.stage != ScreeningStage.inputValidation)
        .toList();
    setState(() {
      _busy = true;
      _failed.removeWhere(
          (image) => image.failure?.stage != ScreeningStage.inputValidation);
      _message = null;
    });
    try {
      await _screen(pending);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _save() async {
    final patientId = _patientId;
    if (patientId == null || _images.isEmpty || _failed.isNotEmpty || _busy) {
      return;
    }
    setState(() {
      _busy = true;
      _message = null;
    });
    try {
      await widget.store.addVisit(
          patientId,
          VisitRecord(
            id: _visitId,
            date: DateTime.now().toUtc(),
            images: _images,
            unavailableEyes: _unavailableEyes,
          ));
      if (!mounted) return;
      widget.onSaved();
      setState(() {
        _images.clear();
        _unavailableEyes.clear();
        _visitId = recordId('VIS');
        _message =
            'Visit saved. Open Patients to view the visit or create a PDF.';
      });
    } catch (_) {
      if (mounted) {
        setState(() => _message =
            'Could not save the visit. Your current selections are still here.');
      }
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  void dispose() {
    if (widget.service == null) _service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text('Retinal image screening',
                style: Theme.of(context).textTheme.headlineSmall,
                softWrap: true),
            const Card(
              child: Padding(
                padding: EdgeInsets.all(12),
                child: Text(
                  'Screening is paused: this build has no validated fundus or image-quality detector. Decodable images are treated as uncertain and will not receive a DR classification.',
                ),
              ),
            ),
            const SizedBox(height: 16),
            FutureBuilder<List<PatientRecord>>(
              future: widget.patients,
              builder: (context, snapshot) {
                if (snapshot.connectionState != ConnectionState.done) {
                  return const LinearProgressIndicator();
                }
                if (snapshot.hasError) {
                  return const Text('Could not load patient records.');
                }
                final patients = snapshot.data ?? [];
                if (patients.isEmpty) {
                  return const Text(
                      'Register a patient before starting a screening visit.');
                }
                return DropdownButtonFormField<String>(
                  isExpanded: true,
                  initialValue: _patientId,
                  decoration: const InputDecoration(labelText: 'Patient'),
                  items: patients
                      .map((p) => DropdownMenuItem(
                          value: p.id,
                          child: Text('${p.name} · ${p.id}',
                              overflow: TextOverflow.ellipsis)))
                      .toList(),
                  onChanged: _busy || _images.isNotEmpty || _failed.isNotEmpty
                      ? null
                      : (id) => setState(() => _patientId = id),
                );
              },
            ),
            const SizedBox(height: 16),
            SegmentedButton<Eye>(
              segments: Eye.values
                  .map((eye) => ButtonSegment(
                        value: eye,
                        label: Text('${eye.label} (${eye.code})'),
                      ))
                  .toList(),
              selected: {_eye},
              onSelectionChanged:
                  _busy ? null : (eyes) => setState(() => _eye = eyes.single),
            ),
            CheckboxListTile(
              contentPadding: EdgeInsets.zero,
              value: _unavailableEyes.contains(_eye),
              title: Text('${_eye.label} unavailable for this visit'),
              subtitle: const Text(
                  'Optional record only; it does not infer a cause or disability.'),
              onChanged: _busy ||
                      _images.any((image) => image.eye == _eye) ||
                      _failed.any((image) => image.eye == _eye)
                  ? null
                  : (value) => setState(() {
                        if (value == true) {
                          _unavailableEyes.add(_eye);
                        } else {
                          _unavailableEyes.remove(_eye);
                        }
                      }),
            ),
            const SizedBox(height: 12),
            Wrap(spacing: 8, runSpacing: 8, children: [
              FilledButton.icon(
                  onPressed: _busy ||
                          _patientId == null ||
                          _unavailableEyes.contains(_eye)
                      ? null
                      : () => _add(camera: false),
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Add fundus images')),
              OutlinedButton.icon(
                  onPressed: _busy ||
                          _patientId == null ||
                          _unavailableEyes.contains(_eye)
                      ? null
                      : () => _add(camera: true),
                  icon: const Icon(Icons.camera_alt_outlined),
                  label: const Text('Camera')),
            ]),
            if (_busy) const LinearProgressIndicator(),
            if (_failed.isNotEmpty) ...[
              const Text(
                  'Unscreened images — retry or remove before saving. No score was recorded for these images.'),
              if (_failed.any((image) =>
                  image.failure?.stage == ScreeningStage.inputValidation))
                const Text(
                    'Input validation is unavailable in this build. Remove these items; retry cannot enable DR classification.'),
              for (final failed in _failed)
                ListTile(
                  title: Text('${failed.eye.code}: ${failed.failure!.message}'),
                  subtitle: Text('Stage: ${failed.failure!.stage.name}'),
                  trailing: IconButton(
                      tooltip: 'Remove failed image',
                      onPressed: _busy
                          ? null
                          : () => setState(() => _failed.remove(failed)),
                      icon: const Icon(Icons.close)),
                ),
              if (_hasRetryableFailures)
                OutlinedButton(
                    onPressed: _busy ? null : _retry,
                    child: const Text('Retry failed images')),
            ],
            if (_message != null)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Semantics(liveRegion: true, child: Text(_message!)),
              ),
            const SizedBox(height: 10),
            for (final eye in Eye.values)
              EyeResults(
                eye: eye,
                images: _images.where((image) => image.eye == eye),
                original: (image) async => File(image.path),
                onRemove: _busy
                    ? null
                    : (image) => setState(() => _images.remove(image)),
              ),
            const SizedBox(height: 12),
            FilledButton.tonal(
              onPressed: _busy ||
                      _patientId == null ||
                      _images.isEmpty ||
                      _failed.isNotEmpty
                  ? null
                  : _save,
              child: const Text('Finalize and save visit'),
            ),
            const Padding(
                padding: EdgeInsets.only(top: 12),
                child: Text(screeningNotice)),
          ],
        ),
      );
}

const screeningNotice =
    'Experimental AI screening only. A result below threshold does not rule out diabetic retinopathy. This output is not a diagnosis or a basis for treatment. Get a qualified eye-care assessment, regardless of the score.';

class EyeResults extends StatelessWidget {
  const EyeResults(
      {super.key,
      required this.eye,
      required this.images,
      required this.original,
      this.onRemove});
  final Eye eye;
  final Iterable<ImageRecord> images;
  final Future<File> Function(ImageRecord) original;
  final ValueChanged<ImageRecord>? onRemove;

  @override
  Widget build(BuildContext context) => Card(
        child: Column(
          children: [
            ListTile(
              title: Text('${eye.label} (${eye.code})'),
              subtitle: Text('${images.length} screened image(s)'),
            ),
            for (final image in images)
              ListTile(
                leading: FutureBuilder<File>(
                  future: original(image),
                  builder: (context, snapshot) => snapshot.hasData
                      ? Image(
                          image: ResizeImage(FileImage(snapshot.data!),
                              width: 96,
                              height: 96,
                              policy: ResizeImagePolicy.fit),
                          width: 48,
                          height: 48,
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) =>
                              const Icon(Icons.broken_image_outlined))
                      : const Icon(Icons.image_outlined),
                ),
                title: Text(
                    '${(image.probability * 100).toStringAsFixed(1)}% DR model probability'),
                subtitle: Text(image.flagged
                    ? 'At or above threshold'
                    : 'Below threshold; DR not ruled out'),
                trailing: onRemove == null
                    ? null
                    : IconButton(
                        tooltip: 'Remove image',
                        onPressed: () => onRemove!(image),
                        icon: const Icon(Icons.close),
                      ),
                onTap: () async {
                  final file = await original(image);
                  if (!context.mounted) return;
                  await showDialog<void>(
                      context: context,
                      builder: (context) => AlertDialog(
                            title: Text('${eye.label} image'),
                            content: Image(
                                image: ResizeImage(FileImage(file),
                                    width: 1200,
                                    height: 1200,
                                    policy: ResizeImagePolicy.fit),
                                fit: BoxFit.contain,
                                errorBuilder: (_, __, ___) =>
                                    const Text('Original image unavailable.')),
                            actions: [
                              TextButton(
                                  onPressed: () => Navigator.pop(context),
                                  child: const Text('Close'))
                            ],
                          ));
                },
              ),
          ],
        ),
      );
}

class _PendingImage {
  _PendingImage(this.file, this.eye) : capturedAt = DateTime.now().toUtc();
  final XFile file;
  final Eye eye;
  final DateTime capturedAt;
  ScreeningFailure? failure;
}
